# Third-party notices

WM3D-WAM is MIT-licensed except where an upstream license applies. Vendored
code remains attributable to its original project and license.

## FastWAM

Files under `src/wm3d_wam/vendor/fastwam/` are derived from FastWAM's Wan2.2
VideoDiT, ActionDiT, MoT, flow scheduler, and support code. They are used under
the FastWAM MIT license in `LICENSES/FASTWAM_LICENSE.txt`.

WM3D-WAM changes are intentionally narrow: package imports, batch-aware
attention masks, optional sparse geometry K/V, and grouped action integration.
The VideoDiT and ActionDiT transformer block implementations are otherwise
kept aligned with the audited upstream code.

Upstream: <https://github.com/AgibotTech/FastWAM>

## GAM / local VGGT-GAM adaptation

Files under `src/wm3d_wam/vendor/vggt_gam/` are derived from the GAM project
and the locally adapted `gam-vggt-prototype`. They are used under the GAM MIT
license in `LICENSES/GAM_LICENSE.txt`.

WM3D-WAM retains the split-and-resume VGGT path and restores the pretrained
camera and point heads required by its geometry contract. The tracking head is
not part of v1.

Upstream: <https://github.com/cvlab-kaist/Geometric-Action-Model>

Audited adaptation: <https://github.com/wxqnl/vggt-gam>

## VGGT

WM3D-WAM does not redistribute VGGT weights or the official VGGT source tree.
At runtime the geometry adapter loads a separately provisioned VGGT source and
checkpoint. Use and redistribution of those materials are governed by Meta's
VGGT License and Acceptable Use Policy, which must be reviewed before assets
are provisioned. A copy of the license from the audited source tree is kept in
`LICENSES/VGGT_LICENSE.txt`.
