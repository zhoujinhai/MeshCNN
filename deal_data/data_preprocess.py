"""
Usage: blender --background --python data_preprocess.py
***注意：先修改stl路径， obj保存路径 和 测试集路径***
"""

import sys
sys.path.append(r"/home/heygears/work/package/hgapi")
import hgapi
# from hgapi import hgapi

import os
import glob
# from vedo import load
from multiprocessing import Process
# from tqdm import tqdm


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


# write 默认只保留5位，这里根据源码编写转换函数 https://vedo.embl.es/_modules/vedo/io.html#write
def stl_2_obj_by_vedo(stl_path, file_out_path):
    """
    stl model convert to obj model
    @objct: stl model path
    @file_out_path: obj保存地址
    """
    from vedo import load
    objct = load(stl_path)
    fr = file_out_path.lower()

    assert fr.endswith(".obj"), "fileoutput 需以obj为后缀"
    outF = open(file_out_path, "w")
    outF.write('# OBJ file format with ext .obj\n')
    outF.write('# File generated by vedo\n')

    for p in objct.points():
        # outF.write("v {:.5g} {:.5g} {:.5g}\n".format(*p))
        outF.write("v {:.10f} {:.10f} {:.10f}\n".format(*p))

    for i, f in enumerate(objct.faces()):
        fs = ''
        for fi in f:
            fs += " {:d}".format(fi + 1)
        outF.write('f' + fs + '\n')

    for l in objct.lines():
        ls = ''
        for li in l:
            ls += str(li + 1) + " "
        outF.write('l ' + ls + '\n')

    outF.close()
    return objct


def stl_2_obj_by_hgapi(stl_path, file_out_path):
    """
    stl model convert to obj model
    @objct: stl model path
    @file_out_path: obj保存地址
    """
    mesh_healing = hgapi.MeshHealing()
    mesh = hgapi.Mesh()
    stl_handler = hgapi.STLFileHandler()
    stl_handler.Read(stl_path, mesh)
    mesh_healing.RemoveSmallerIslands(mesh)  # 删除孤立块
    mesh_healing.RemoveDanglingFace(mesh)  # 删除悬边
    obj_handler = hgapi.OBJFileHandler()
    obj_handler.Write(file_out_path, mesh)


def stl_2_obj_batch(stl_list, obj_save_path):
    """
    stl格式批量转换为obj格式
    """
    # for i in tqdm(range(len(stl_list))):
    for i in range(len(stl_list)):
        stl_path = stl_list[i]
        print(stl_path)
        save_name = os.path.basename(stl_path).replace(".stl", ".obj")
        obj_path = os.path.join(obj_save_path, save_name)
        # stl_2_obj_by_vedo(stl_path, obj_path)
        stl_2_obj_by_hgapi(stl_path, obj_path)
        # print("The {} model is converted!".format(i))


def parallel_stl_2_obj(stl_list, obj_save_path, n_workers=8):
    """
    多进程处理
    """
    if len(stl_list) < n_workers:
        n_workers = len(stl_list)

    assert n_workers > 0, "没有要处理的数据，请检查路径"
    chunk_len = len(stl_list) // n_workers

    chunk_lists = [stl_list[i:i + chunk_len] for i in range(0, (n_workers - 1) * chunk_len, chunk_len)]
    chunk_lists.append(stl_list[(n_workers - 1) * chunk_len:])

    process_list = [Process(target=stl_2_obj_batch, args=(chunk_list, obj_save_path)) for chunk_list in chunk_lists]
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()


if __name__ == "__main__":

    stl_dir = "/run/user/1000/gvfs/smb-share:server=10.99.11.210,share=meshcnn/Test_5044/modify/stl"
    obj_save_dir = "/run/user/1000/gvfs/smb-share:server=10.99.11.210,share=meshcnn/Test_5044/modify/obj"
    test_dir = "/run/user/1000/gvfs/smb-share:server=10.99.11.210,share=meshcnn/Test_5044/modify/down_obj"
    target_faces = 5000

    if not os.path.isdir(obj_save_dir):
        os.makedirs(obj_save_dir)
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)

    stl_list = glob.glob(os.path.join(stl_dir, "*.stl"))
    print("need deal {} data".format(len(stl_list)))
    #
    # #  stl to obj
    parallel_stl_2_obj(stl_list, obj_save_dir, 8)
    #
    import bpy
    # # 下采样
    print('args: ', obj_save_dir, target_faces, test_dir)
    blender = SampleProcess(obj_save_dir, target_faces, test_dir)
