# RefMods for MiniMax H3 — "How do I…?"

The companion to the README's RefMod section. A RefMod is your reference photos (or clips) saved as one small file that the [ComfyUI-MiniMaxH3Mod](https://github.com/shingo257/comfyui-minimaxh3mod) nodes (by **@shingo257**, with a mod library and guide from **@malcolmrey**) load like a LoRA and feed to H3's reference path. No training run, no captions needed for the basic kind, a file in minutes. Fizgig makes them, and adds the one thing no other maker has: it can tune the file against the H3 model itself so the person survives shots your photos never showed.

Two places in Fizgig do RefMod work:

- **Training tab → Base Model → MiniMax H3 RefMod** makes the file.
- **RefMod Studio** (its own tab) tests it before it goes anywhere near ComfyUI.

---

## How do I make a RefMod?

1. On the **Start** tab, point Fizgig at a folder of photos of the person (or clips — see "clips" below). The usual Image Prep face crops are ideal: the face large in frame, consistent quality.
2. On the **Training** tab, pick **MiniMax H3 RefMod** in the Base Model selector. The tab shrinks to Output plus one card.
3. Pick a preset from **Load Preset** (see the next answer). The card fills in.
4. Set the **Output Directory** to your ComfyUI `models/refmods` folder. It is remembered per model family, so you only do this once, and every mod lands ready to load.
5. Press **Start**. A plain encode takes seconds after caching. An optimised one takes a few minutes on a 32 GB card.

The file is `<name>.safetensors` in that folder. Load it in ComfyUI with **Load H3 RefMods → Apply H3 RefMod**, the same as any other mod.

## Which preset do I pick?

**For a person:**

| Preset | What it makes | When |
|---|---|---|
| **Community recipe** (the default) | A plain encode of 8 photos at 1 MP — what the pack's own extractor makes, at the popular library's budget. | The quick one. No captions needed. |
| **Fizgig recipe lite** | 16 photos at 0.5 MP, optimised against the model. | The one to reach for: about the same token cost at generation as the library's files, with the optimisation. |
| **Fizgig recipe** | The same at 1 MP. | A heavier file that is slower to make and to generate with, for a little more sharpness, smoother skin and a small edge in likeness over lite. |

**For a look rather than a person:**

| Preset | What it makes | When |
|---|---|---|
| **Style, community** | The pack's concept recipe: every still pooled to a small 16×16 grid, plain encode. | A palette, a grade, a feel. Light enough to stack beside a character mod. |
| **Style, high fidelity** | 8 stills at 1 MP, full resolution, plain encode. | When the look lives in texture, grain or brushwork. |

The style presets are plain encodes and need no captions. The optimiser has not been measured on a style set, so leave Steps at 0 there unless you are experimenting.

## Do I need captions?

Only if Steps is above 0. A plain encode (Steps 0, which the community and style presets use) reads the photos alone, and the text-encoder step is skipped. The optimised presets train against your captioned stills, so caption the folder first, the Captions tab with a trigger word is enough.

## What does "Steps" actually do, and is it worth it?

A plain encode makes H3 copy your photos, and it copies them well as long as the shot looks like the photos. With Steps above 0, the H3 model is loaded and frozen, and the file itself is tuned so that H3 reproduces the person from it: the ordinary training loss on your stills, held close to the plain encode and worked only at the noise levels that matter.

What that buys, measured on same-seed clips against the dataset:

- On a shot like the photos, the plain encode and the optimised mod score level on likeness.
- On a look the photos never showed (the same face under clown make-up), the optimised mod scored 28 against 21, and its worst frame was above the plain encode's best. It carries the person, not the pictures.
- In both cases the optimised mod renders skin with visibly less noise and fine detail sharper. A plain encode carries the photo's grain and the VAE's own error, and H3 copies all of it; the steps keep what helps the model predict the subject and let the rest go.

200 steps is the default and takes a few minutes on a 32 GB card. Each step rides one reference beside one still, so it is quick whatever the reference count, and it runs on 16 and 24 GB cards without streaming.

## Which H3 model is the mod for?

Both are covered, and it matters. **Training Base** on the card picks the model the mod is optimised against. Selecting the RefMod family switches it to **Reference (ref2va)**, the model built to take references; pick **First / Last Frame (fl2va)** to make one for that model instead.

Measured: the same 16 stills optimised for ref2va and run on the first/last-frame model scored 60 against the dataset; optimised for fl2va and run there, 65, with the matched mod's worst frame above the mismatched mod's best. Made for the model it runs on, a mod holds the person about as well on either. So: load the mod in ComfyUI with the model it was made for, and make one per model if you use both.

## What do the controls on the card mean?

- **Grid.** Full reference keeps every photo at its size, the only setting that carries a face. The pooled grids (32×32, 16×16, 8×8) make small stackable concept mods.
- **References.** How many stills go in, photos first, then clips. 16 is the measured default for a person. A live token count sits beside it against the pack's 5,120 default cap: more references or bigger ones cost time and VRAM at every generation.
- **Steps.** Above.
- **Description / Concept.** The mod's hint: a few words saying what it is ("a ginger woman with messy hair", "a 1970s film look") and what kind, in the pack's terms (identity, style, pose_motion, clothing, background, generic). Both go into the file. RefMod Studio's **+ mod hints** button and the pack's loader put "concept: description" into the prompt, which the pack's own guide says a mod needs to anchor on.
- **Clips.** How a clip in your folder enters the mod: **as its sharpest still** (one frame, the face, the identity choice) or **as motion** (every latent frame of the clip, the pack's video reference, for a dance or a camera move). Tokens are per frame, so a clip costs its length. Prepare clips with Gizmo first, cut to H3's frame grid at the right size.
- **Target MP.** The resolution the references are encoded at. With Steps above 0 it sizes the references only; the optimiser's own stills are always cached at 0.25 MP in a second, lighter pass.

Crops keep the face: the references are placed on one canvas (the majority aspect, sized by the middle reference), and any photo that has to be cropped to fit is cropped around its face rather than the frame centre, never tighter than the canvas needs.

## How do I test a mod before ComfyUI?

Open **RefMod Studio**. Top to bottom:

1. **Setup.** Point it at your mods folder (ComfyUI's `models/refmods`), pick the model (Reference or First / Last Frame, the same choice you make in ComfyUI) and the base precision. Load base once per session; Render does it for you if you skip it.
2. **Mods.** One mod per row: tick, the mod, its Strength, its Copies. Type in the picker to narrow a long list. The button underneath adds a second row so you can see two mods together, a character and a style, each at its own strength. That is for trying things; Bake writes one mod per file.
3. **Apply.** How the mods are applied while the picture renders. Each block names the node setting it maps to and says what it does: Strength (Retention) is a master over every row; Shuffle picks which reference leads; Fade across the clip is for video mods; Change during the render eases the reference off for the final texture steps, or in for the identity at the end. Nothing here edits a mod file.
4. **Render.** Set the prompt (press **+ mod hints** to append each mod's hint), seed, length, size, steps and Turbo, then **Render**. With mods appears on the left the moment it is done, No mod on the right for comparison. **Render sweep** steps one dial through its useful values and gives you a row of 22-frame clips to compare; click a chip to play it.
5. **Actions.** **ComfyUI settings** copies the exact values for the pack's nodes. **Bake as new RefMod** writes a new file with one row's Strength, the master Strength, its Copies and the frame curve folded in, so it loads plainly at 1.0. Save preview keeps the picture; Save and Load setup keep the whole tab.

The numbers are the pack's own maths, so what you see in the Studio is what the nodes will do.

## Can I bake two mods into one file?

No, and the Studio no longer offers it. A file holds one reference block, and the model reads a block as one subject, so two mods written into one file come out as one of them. Two mods together are two loader slots in ComfyUI. The ComfyUI settings readout gives the values for both.

## Why does my mod change the framing or the jewellery?

More references, especially at 1 MP, pull the generation toward the photos: their framing, their earrings. That is the reference count, not the optimisation. Use the Strength (Retention) dial in the Studio, or fewer references, or the pack's usual habit of running stacked mods below 1.0.

## Why is likeness low on a clip where she looks away?

The likeness meter scores against frontal reference crops, so a turned head costs ten points or more on its own. Judge on frontal stretches, or prompt "facing the camera" for the comparison.

## Where is the file, and is it really standard?

In the RefMod family's Output folder as `<name>.safetensors`. It is a standard RefMod: the pack's own file layout, loadable by its nodes with nothing special. Fizgig adds a few of its own keys to the header that the nodes ignore.
