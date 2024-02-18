import torch
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer.blending import softmax_rgb_blend
from pytorch3d.renderer import (
    TexturesVertex,
    look_at_view_transform,
    FoVOrthographicCameras,
    PointLights,
    DirectionalLights,
    AmbientLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    SoftGouraudShader,
    SplatterPhongShader,
    SoftSilhouetteShader,
    HardFlatShader,
    BlendParams
)


def get_renderer(
        resolution,
        aa_factor,
        device,
        lights):

    cam_R, cam_T = look_at_view_transform(
        eye=[(0, 0, 1.)],
        at=((0, 0, 0), ),
        up=((0, 1, 0),),
    )
    cameras = FoVOrthographicCameras(
        znear=0.,
        zfar=2.,
        max_y=1.,
        min_y=-1.,
        max_x=1.,
        min_x=-1.,
        R=cam_R,
        T=cam_T
    ).to(device)

    raster_settings = RasterizationSettings(
        image_size=resolution * aa_factor,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    ).to(device)

    return renderer


def normal_shading(meshes, fragments) -> torch.Tensor:
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    # vertex_normals = (vertex_normals+1.) * 255/2.
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )

    return pixel_normals


class SoftNormalShading(ShaderBase):
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = normal_shading(
            meshes=meshes,
            fragments=fragments,
        )
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(
            colors, fragments, blend_params, znear=znear, zfar=zfar
        )
        return images


def get_normal_renderer(
        resolution,
        device
    ) -> None:
    cam_R, cam_T = look_at_view_transform(dist=1.0, elev=0, azim=180)
    cameras = FoVOrthographicCameras(
        znear=0.,
        zfar=2.,
        max_y=1.,
        min_y=-1.,
        max_x=1.,
        min_x=-1.,
        R=cam_R,
        T=cam_T
    ).to(device)

    raster_settings = RasterizationSettings(
        image_size=resolution,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftNormalShading(
            cameras=cameras,
            blend_params=BlendParams(
                background_color=(0., 0., 0.)
            )
        ),
    ).to(device)

    return renderer
