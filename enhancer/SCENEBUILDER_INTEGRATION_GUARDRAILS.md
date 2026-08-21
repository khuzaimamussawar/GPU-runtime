# SceneBuilder2 enhancer integration guardrails

Status: **mandatory companion to `enhancer/PLAN.md` for SceneBuilder2-side implementation**

This document exists to keep the SceneBuilder integration narrow, backward-compatible and non-destructive. Where older SceneBuilder-side enhancer wording conflicts with these guardrails, these guardrails win for the SceneBuilder integration.

The implementation goal is **read old projects safely, write new projects cleanly, and never require a destructive migration of existing `project_video_timeline.data_json`.**

---

## 1. Non-destructive rollout is a hard requirement

Do not run a bulk rewrite, backfill, rename or destructive normalization over existing `project_video_timeline` rows merely to make enhancer media naming cleaner.

Forbidden rollout behavior:

```text
ALTER/rename existing Director media fields
mass UPDATE existing project_video_timeline.data_json
clear legacy videoUrl/renderVideoUrl values
replace existing source/generated media pointers with upscale pointers
delete legacy H3 preview objects
rewrite every H.264 URL to H.265
write normalized state merely because a project was opened/read
```

Existing projects must remain usable before, during and after enhancer rollout.

Compatibility rule:

```text
READ OLD
  -> resolve legacy/current media shapes in memory
  -> do not mutate D1 just because the row is old

WRITE NEW
  -> use the clean provider-aware contract for new generation/upscale results
```

A normal explicit user save/edit may persist additive enhancer fields, but it must preserve unrelated and unknown segment keys. Loading a project alone must not trigger a compatibility rewrite.

---

## 2. SceneBuilder2 implementation scope must stay narrow

Primary files/surfaces allowed to change for this integration:

```text
src/components/VideoGenerationTimeline.tsx
vite.director-master-render.ts
existing enhancer generation/control-plane routes and writeback helpers
existing Director tests / enhancer tests
vite.config.ts only when an existing guarded patch must be registered/folded
```

Do not refactor unrelated working Director behavior merely because the enhancer work touches the same component.

Preserve existing working behavior for:

```text
Director geometry / trim / split / linked-unit behavior
Audio Studio timing and audio reference logic
prompt generation / enhancement
H3 generation controls and lifecycle
Grok / Seedance generation controls
Storyboard source/restore behavior
selection behavior
keyboard/playback behavior
R2 ownership/reference accounting
project revision/save conflict behavior
current Topaz behavior except where the pre-upscale input invariant specifically requires a safe source resolver
```

If a change is not required for Source / Generated / Upscaled selection, authoritative generated media resolution, no-recursive-upscale input, enhancer queueing or final render handoff, leave it alone.

---

## 3. Do not stack conflicting Director media resolvers

SceneBuilder2 currently uses guarded Vite source transforms around Director media selection and final-render handoff.

The implementation must have **one canonical set of media resolver semantics**. Do not let two Vite plugins or one source implementation plus a stale Vite transform independently reinterpret the same media fields.

Required approach:

```text
preferred:
  fold the final media-resolution behavior into the existing canonical Director source/patch path

acceptable:
  update source and update/remove the old guarded transform in the same change

forbidden:
  leave old auto-upscale/master logic active while adding a second resolver elsewhere
```

Guarded transforms must keep exact-match failure behavior. If the source underneath changes and the transform no longer matches exactly, the build should fail rather than silently ship a half-applied patch.

Before merge, verify all registered Director Vite transforms still match the current `VideoGenerationTimeline.tsx` and `TimelineRender.tsx` source.

---

## 4. Logical media variants, not codec guesses

Every Director segment may logically expose:

```text
SOURCE
GENERATED
UPSCALED
```

These are product roles, not codec names.

### SOURCE

Resolve actual source media from persisted/current Director state:

```text
actual uploaded source video exists and is the selected source
  -> sourceVideoUrl / sourceVideoObjectKey

otherwise current image source exists
  -> imageUrl / imageObjectKey

otherwise storyboard source reconstructed by the existing Director load path
  -> current storyboard image
```

Never use these as final source media:

```text
thumbnailUrl
videoThumbnailUrl
originalThumbnailUrl
```

`originalImageUrl` remains restore/reset metadata. It is not allowed to silently replace a newer `imageUrl` selected/uploaded by the user.

Historical `mediaSource` values are hints/state, not proof that every corresponding pointer exists. In particular, legacy rows may say `mediaSource='video-uploaded'` without a current `sourceVideoUrl`; resolver logic must inspect the real populated pointers and preserve current source selection behavior.

### GENERATED

Resolve the **authoritative original generated asset before enhancer**.

Do not define Generated as `upscaledVideoUrl`.

### UPSCALED

Resolve only a successfully written enhancer derivative:

```text
upscaledVideoUrl / upscaledVideoObjectKey
```

It is optional and never replaces/deletes Source or Generated.

---

## 5. Provider-aware generated-media contract

H.264 does not mean preview.

Canonical generated roles:

| Generator | Editor/browser generated media | Authoritative original generated media |
|---|---|---|
| H3 | H.264 preview | H.265 generated master |
| Grok Imagine | H.264 generated result | same H.264 generated result |
| Seedance | H.264 generated result | same H.264 generated result |

Only the canonical H3 preview identity is treated as a preview:

```text
/video/previews/NAME-h264-preview.mp4
```

Its corresponding H3 master identity is:

```text
/video/generated/NAME-h265.mp4
```

Never implement:

```text
if codec/url contains H264 -> rewrite to H265
```

That would break Grok, Seedance, uploaded video and future providers whose authoritative result is H.264.

---

## 6. Legacy H3 projects stay stored as-is

Existing H3 projects may contain this valid legacy compatibility shape:

```text
videoUrl
  -> H3 H.264 preview

renderVideoUrl
  -> sometimes the same H3 H.264 preview

renderVideoObjectKey
  -> H3 H.265 generated master
```

Do not repair these rows with a mandatory D1 migration.

Authoritative generated resolution order for H3/current legacy compatibility:

```text
1. valid renderVideoObjectKey
   -> resolve/use that exact master object

2. renderVideoUrl when it is not the canonical H3 preview
   -> use it

3. canonical H3 preview URL/key
   -> derive the canonical generated H.265 master path
   -> use only the master path; never intentionally send the preview to enhancer/final render

4. provider single-result fallback
   -> use videoUrl only when it is not a canonical H3 preview or provider semantics prove it is the authoritative single result
```

For a confirmed H3 canonical preview, this compatibility mapping is allowed:

```text
/video/previews/NAME-h264-preview.mp4
        ->
/video/generated/NAME-h265.mp4
```

But `renderVideoObjectKey` is the stronger signal when it already identifies the H.265 master.

If an H3 master cannot be resolved, keep the preview usable in the editor if it already works there, but do **not** silently upscale/final-render the preview as though it were the master. Surface a master-missing/ineligible error and leave the existing project media untouched.

---

## 7. Clean writeback for all new generations

New H3 completion must write clean separate preview/master roles:

```text
videoUrl             = H3 H.264 preview URL
videoObjectKey       = H3 H.264 preview key
renderVideoUrl       = H3 H.265 master URL
renderVideoObjectKey = H3 H.265 master key
```

New Grok/Seedance completion has one authoritative H.264 result, so both roles may deliberately point at the same object:

```text
videoUrl             = provider H.264 result URL
videoObjectKey       = provider H.264 result key
renderVideoUrl       = SAME H.264 result URL
renderVideoObjectKey = SAME H.264 result key
```

Do not create a useless H.265 duplicate merely to make the fields look symmetrical.

Future provider rule:

```text
one generated deliverable
  -> video* and renderVideo* may be the same object

separate preview + master deliverables
  -> video* = editor/browser asset
  -> renderVideo* = authoritative original generated asset
```

---

## 8. One canonical resolver family

SceneBuilder should conceptually have one resolver family used by preview, upscale input and final-render handoff:

```text
resolveSourceMedia(segment)
resolveGeneratedEditorMedia(segment)
resolveGeneratedAuthoritativeMedia(segment)
resolveUpscaledMedia(segment)
resolveActiveDirectorMedia(segment)
resolvePreUpscaleVideo(segment/unit)
```

Do not duplicate provider/master logic independently in:

```text
Director preview
Upscale Selected
Upscale All
Re-upscale
TimelineRender
Topaz path
local enhancer path
```

All callers should consume the same authoritative result.

`TimelineRender` receives a resolved clip DTO. It must not query/reinterpret D1 media pointers itself.

---

## 9. Existing per-clip Show Source UI stays

Do not replace the working per-clip `Show source image/video` interaction with a large new selector just to expose internal state.

Internally:

```ts
activeVisualView?: 'source' | 'generated' | 'upscaled';
```

Normal UI remains based on the existing Source/result interaction.

When a successful upscale exists, the existing Upscale controls add only the actions needed to make the third state reversible:

```text
Use Upscaled
Undo Upscale
Re-upscale
```

Behavior:

```text
Upscaled active + Undo Upscale
  -> return to upscaledFromView (source or generated)

Upscaled stored but inactive + Use Upscaled
  -> activeVisualView = upscaled

Show Source / Show Generated
  -> continue using the existing Director interaction
```

Do not delete Source/Generated media when switching views.

Legacy rows whose old code displayed an upscale while `activeVisualView='generated'` must preserve visible behavior through compatibility normalization/resolution. Do not write a migration solely to rename that state.

---

## 10. Re-upscale can never chain an upscale

This is a hard SceneBuilder request-building invariant.

```text
active source
  -> uploaded/source video

active generated
  -> authoritative original generated asset

active upscaled + upscaledFromView=source
  -> original uploaded/source video

active upscaled + upscaledFromView=generated
  -> authoritative original generated asset
```

Never:

```text
upscaledVideoUrl -> enhancer -> second upscale
```

This applies to:

```text
single Re-upscale
Upscale Selected
Upscale All / Re-upscale All
retry
stale replacement
Topaz request construction
local enhancer request construction
admin/manual request construction
```

For old upscale records without durable origin metadata, infer only from retained pre-upscale assets:

```text
authoritative generated asset exists -> generated
else actual source video exists       -> source
else                                  -> do not queue recursive video upscale
```

---

## 11. `pending_upscales` is not long-term media truth

Completed/failed `pending_upscales` jobs are short-lived and may be deleted after about one hour.

While a job exists it should record the exact execution input:

```text
source_view
source URL
source object key
source trim/range
Director speed
timing/VFI settings
```

That is short-term operational evidence only.

Long-term behavior must come from the Director segment JSON:

```text
activeVisualView
upscaledVideoUrl
upscaledVideoObjectKey
upscaledFromView
upscaledSourceUrl
upscaledSourceObjectKey
upscaledSourceTrimInMs
upscaledSourceTrimOutMs
upscaledSourceSpeed
upscaledTimingBaked
upscaledTargetFps
upscaledOutputDurationMs
```

Months later, Undo/Re-upscale/staleness checks must still work after the operational job row is gone.

---

## 12. Upscale writeback is additive and atomic

First successful upscale:

```text
retain current source fields
retain current generated fields
write upscaledVideoUrl / upscaledVideoObjectKey
write durable base origin/source identity/timing
set activeVisualView = upscaled
```

Re-upscale:

```text
keep old successful upscale active/selectable
queue replacement from original source/generated base
verify new output
atomically replace upscale pointer + provenance
keep activeVisualView = upscaled when it was active
```

Failure/cancel must leave the previous successful upscale intact.

Never overwrite:

```text
sourceVideoUrl
imageUrl
videoUrl
renderVideoUrl
renderVideoObjectKey
```

with the enhancer derivative.

---

## 13. Final TimelineRender input contract

Video Generation Timeline resolves each segment before handoff.

Final render may receive any mixture in one project:

```text
current storyboard/manual image source
uploaded original video
Grok H.264 original generated result
Seedance H.264 original generated result
H3 H.265 original generated master
successful upscaled derivative
```

It must never receive a thumbnail as media.

It must never receive the canonical H3 H.264 preview when the H3 master exists.

The resolved DTO remains:

```text
url
sceneDuration
startTimeOffset
speed
```

Timing-baked upscale rules from the canonical plan still apply. Do not mutate persisted Director `segment.speed` to make an enhanced file render correctly.

---

## 14. Current-project compatibility tests are mandatory

Do not commit a real customer/project export as a public fixture. Build synthetic fixtures that reproduce the audited shapes.

Required fixtures/tests:

```text
legacy H3:
  videoUrl = H.264 preview
  renderVideoUrl = same H.264 preview
  renderVideoObjectKey = H.265 master
  -> editor preview unchanged
  -> enhancer/final render resolve H.265 master
  -> no D1 rewrite required

clean new H3:
  video* = H.264 preview
  renderVideo* = H.265 master

Grok:
  video* == renderVideo* == H.264 authoritative result

Seedance:
  video* == renderVideo* == H.264 authoritative result

uploaded source + generated result on same segment
image source with dedicated thumbnail
image source where thumbnail falls back to imageUrl
old upscale visible through legacy generated semantics
new upscale from source
new upscale from generated
stale upscale after source/generated replacement
missing H3 master
```

Hard assertions:

```text
opening/reading an old project performs no compatibility D1 write
unknown segment fields survive save/hydration
legacy H3 preview URL is never submitted when master identity exists
H.264 codec alone never triggers H3 conversion
Grok/Seedance H.264 originals remain valid
thumbnail fields never become final render/upscale media
Re-upscale never submits upscaledVideoUrl
Undo does not delete any media
failed replacement does not remove previous upscale
mixed old/new segments render in one timeline
```

---

## 15. Rollout order minimizes blast radius

Implement in this order:

```text
1. resolver unit tests + legacy/provider fixtures
2. provider-aware authoritative generated resolver
3. no-recursive pre-upscale resolver
4. TimelineRender handoff uses canonical resolver
5. additive activeVisualView='upscaled' + durable provenance hydration
6. Undo / Use Upscaled / Re-upscale on existing UI
7. local enhancer queue wiring
8. bulk Upscale Remaining / Re-upscale All
9. only after all above: optional cleanup/backfill discussion
```

There is **no required cleanup/backfill** for production launch.

A later optional normalization of legacy `renderVideoUrl` may be considered only after the runtime resolver is proven in production. It must be a separate reviewed operation, never bundled into enhancer rollout, and correctness must not depend on it.

---

## 16. Merge/review stop conditions

Do not merge SceneBuilder integration if any of these are true:

```text
a code path still auto-prefers upscaledVideoUrl as Generated
getOriginalVideoForUnit or equivalent can return upscaledVideoUrl
H3 final/upscale input can select canonical H.264 preview when master exists
Grok/Seedance H.264 results are treated as previews
load normalization writes old rows merely because they are old
existing Show Source behavior is replaced/broken unnecessarily
unrelated Director geometry/audio/prompt logic changed without a required reason
a Vite guarded transform no longer matches exactly
current-project compatibility fixtures fail
```

The safety principle is simple:

```text
old projects: resolve, do not rewrite
new generation: write clean provider-aware roles
upscale: add a derivative, never destroy the base
final render: consume one already-resolved authoritative asset
```
