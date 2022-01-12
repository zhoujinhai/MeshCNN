import bpy
import os
import sys
import glob

'''
Simplifies mesh to target number of faces
Requires Blender 2.8
Author: Rana Hanocka

@input: 
    <obj_file>
    <target_faces> number of target faces
    <outfile> name of simplified .obj file

@output:
    simplified mesh .obj
    to run it from cmd line:
    /opt/blender/blender --background --python down_sample.py /home/heygears/work/MeshCNN/datasets/tooth/obj 5000 /home/heygears/work/MeshCNN/datasets/tooth/down_obj_1
'''

class SampleProcess:
    def __init__(self, obj_path, target_faces, export_path):
        obj_list = glob.glob(os.path.join(obj_path, "*.obj"))
        print(len(obj_list))
        for i, obj in enumerate(obj_list):
            print(i)
            mesh = self.load_obj(obj)
            save_name = os.path.basename(obj)
            export_name = os.path.join(export_path, save_name)
            self.simplify(mesh, target_faces)
            self.export_obj(mesh, export_name)


    def load_obj(self, obj_file):
        bpy.ops.import_scene.obj(filepath=obj_file, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl", use_edges=True,
                                 use_smooth_groups=True, use_split_objects=False, use_split_groups=False,
                                 use_groups_as_vgroups=False, use_image_search=True, split_mode='ON')
        ob = bpy.context.selected_objects[0]
        return ob

    def subsurf(self, mesh):
        # subdivide mesh
        bpy.context.view_layer.objects.active = mesh
        mod = mesh.modifiers.new(name='Subsurf', type='SUBSURF')
        mod.subdivision_type = 'SIMPLE'
        bpy.ops.object.modifier_apply(modifier=mod.name)
        # now triangulate
        mod = mesh.modifiers.new(name='Triangluate', type='TRIANGULATE')
        bpy.ops.object.modifier_apply(modifier=mod.name)

    def simplify(self, mesh, target_faces):
        bpy.context.view_layer.objects.active = mesh
        mod = mesh.modifiers.new(name='Decimate', type='DECIMATE')
        bpy.context.object.modifiers['Decimate'].use_collapse_triangulate = True
        #
        nfaces = len(mesh.data.polygons)
        if nfaces < target_faces:
            self.subsurf(mesh)
            nfaces = len(mesh.data.polygons)
        ratio = (target_faces+0.5) / float(nfaces)
        mod.ratio = float('%s' % ('%.6g' % (ratio)))
        print('faces: ', mod.face_count, " nfaces: ", nfaces, " ratio: ", mod.ratio, " down_faces: ", nfaces*mod.ratio)
        bpy.ops.object.modifier_apply(modifier=mod.name)

    def export_obj(self, mesh, export_name):
        outpath = os.path.dirname(export_name)
        if not os.path.isdir(outpath): os.makedirs(outpath)
        print('EXPORTING', export_name)
        bpy.ops.object.select_all(action='DESELECT')
        mesh.select_set(state=True)
        bpy.ops.export_scene.obj(filepath=export_name, check_existing=False, filter_glob="*.obj;*.mtl",
                                 use_selection=True, use_animation=False, use_mesh_modifiers=True, use_edges=True,
                                 use_smooth_groups=False, use_smooth_groups_bitflags=False, use_normals=True,
                                 use_uvs=False, use_materials=False, use_triangles=True, use_nurbs=False,
                                 use_vertex_groups=False, use_blen_objects=True, group_by_object=False,
                                 group_by_material=False, keep_vertex_order=True, global_scale=1, path_mode='AUTO',
                                 axis_forward='-Z', axis_up='Y')

obj_file_path = sys.argv[-3]
target_faces = int(sys.argv[-2])
export_path = sys.argv[-1]

print('args: ', obj_file_path, target_faces, export_path)
blender = SampleProcess(obj_file_path, target_faces, export_path)
