import importlib.util
import threading
import time
import unittest
from pathlib import Path
from unittest import mock


SPEC = importlib.util.spec_from_file_location("render_fast_server", Path(__file__).with_name("server.py"))
assert SPEC and SPEC.loader
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)


class RenderFastSchedulerTests(unittest.TestCase):
    def test_parallel_worker_profile_table(self) -> None:
        expected = {
            6: {(3840, 2160, 30): 3, (3840, 2160, 48): 2, (3840, 2160, 60): 1, (2560, 1440, 48): 4, (2560, 1440, 60): 3, (1920, 1080, 48): 6},
            9: {(3840, 2160, 30): 4, (3840, 2160, 48): 3, (3840, 2160, 60): 2, (2560, 1440, 48): 6, (2560, 1440, 60): 5, (1920, 1080, 48): 9},
            12: {(3840, 2160, 30): 6, (3840, 2160, 48): 4, (3840, 2160, 60): 3, (2560, 1440, 48): 8, (2560, 1440, 60): 7, (1920, 1080, 48): 12},
            16: {(3840, 2160, 30): 8, (3840, 2160, 48): 5, (3840, 2160, 60): 4, (2560, 1440, 48): 10, (2560, 1440, 60): 9, (1920, 1080, 48): 12},
        }
        for vcpus, profiles in expected.items():
            for (width, height, fps), workers in profiles.items():
                with self.subTest(vcpus=vcpus, width=width, height=height, fps=fps):
                    settings = {
                        "_physicalVcpus": vcpus,
                        "_cpuBudget": max(1, int(vcpus * 0.90)),
                        "width": width,
                        "height": height,
                        "fps": fps,
                    }
                    self.assertEqual(SERVER.parallel_clip_worker_count(settings), workers)

    def test_non_reference_vcpu_counts_use_the_same_formula(self) -> None:
        expected_4k48 = {8: 3, 10: 3, 14: 4, 20: 6}
        for vcpus, workers in expected_4k48.items():
            with self.subTest(vcpus=vcpus):
                settings = {
                    "_physicalVcpus": vcpus,
                    "_cpuBudget": max(1, int(vcpus * 0.90)),
                    "width": 3840,
                    "height": 2160,
                    "fps": 48,
                }
                self.assertEqual(SERVER.parallel_clip_worker_count(settings), workers)

    def test_future_clip_blocks_without_overtaking_current_clip(self) -> None:
        buffer = SERVER.OrderedFrameBuffer(2, 7)
        buffer.put(1, b"future", lambda: None)
        buffer.close(1)
        buffer.put(0, b"current", lambda: None)
        buffer.close(0)
        self.assertEqual(buffer.take(0, lambda: None), b"current")
        self.assertEqual(buffer.take(1, lambda: None), b"future")

    def test_per_clip_byte_limit_backpressures_producer(self) -> None:
        buffer = SERVER.OrderedFrameBuffer(1, 1)
        buffer.put(0, b"a", lambda: None)
        started = threading.Event()

        def produce() -> None:
            started.set()
            buffer.put(0, b"b", lambda: None)
            buffer.close(0)

        producer = threading.Thread(target=produce)
        producer.start()
        started.wait(1)
        time.sleep(0.05)
        self.assertTrue(producer.is_alive())
        self.assertEqual(buffer.take(0, lambda: None), b"a")
        producer.join(1)
        self.assertFalse(producer.is_alive())
        self.assertEqual(buffer.take(0, lambda: None), b"b")

    def test_memory_headroom_does_not_block_below_hard_ceiling(self) -> None:
        with mock.patch.object(SERVER, "system_memory_stats", return_value={"systemMemoryAvailable": True, "systemMemoryPercent": 89.9}), mock.patch.object(SERVER.time, "sleep") as sleep:
            SERVER.wait_for_system_memory_headroom(lambda: self.fail("should not stop below the memory ceiling"))
        sleep.assert_not_called()

    def test_decoder_start_failure_closes_its_ordered_queue(self) -> None:
        buffer = SERVER.OrderedFrameBuffer(1, 1024)
        job = {"jobId": "test-job"}
        clip = {"type": "video", "url": "https://example.invalid/source.mp4", "sceneDuration": 1, "speed": 1}
        settings = {"width": 2, "height": 2, "fps": 1, "_ffmpegThreads": 1}
        with mock.patch.object(SERVER.subprocess, "Popen", side_effect=FileNotFoundError("ffmpeg unavailable")):
            SERVER.prepare_visual_unit(job, 0, clip, settings, 6, 1, buffer, threading.Event(), 0)
        self.assertIsNone(buffer.take(0, lambda: None))
        self.assertIn("ffmpeg unavailable", buffer.error(0) or "")

    def test_visual_unit_requests_exact_cfr_frame_budget(self) -> None:
        buffer = SERVER.OrderedFrameBuffer(1, 1024)
        job = {"jobId": "test-job"}
        clip = {"type": "video", "url": "https://example.invalid/source.mp4", "sceneDuration": 1.5, "speed": 1}
        settings = {"width": 2, "height": 2, "fps": 48, "_ffmpegThreads": 1}
        process = mock.Mock()
        process.stdout.read.side_effect = [b"",]
        process.stderr.read.return_value = b""
        process.wait.return_value = 0
        process.poll.return_value = 0
        with mock.patch.object(SERVER.subprocess, "Popen", return_value=process) as popen:
            SERVER.prepare_visual_unit(job, 0, clip, settings, 6, 1, buffer, threading.Event(), 0)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("-fps_mode") + 1], "cfr")
        self.assertEqual(command[command.index("-r") + 1], "48")
        self.assertEqual(command[command.index("-frames:v") + 1], "72")
        self.assertIn("tpad=stop_mode=clone:stop_duration=0.020833333333333332", command[command.index("-vf") + 1])


if __name__ == "__main__":
    unittest.main()
