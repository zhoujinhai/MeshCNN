import vtk
from vtkplotter import *
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import os


def get_gum_line_pts(gum_line_path):
    """
    读取牙龈线文件，pts格式
    """
    f = open(gum_line_path)
    pts = []

    is_generate_by_streamflow = False  # 是否由前台界面生成

    for num, line in enumerate(f):
        if 0 == num and line.strip() == "BEGIN_0":
            is_generate_by_streamflow = True
            continue
        if line.strip() == "BEGIN" or line.strip() == "END":
            continue
        if is_generate_by_streamflow:
            line = line.strip()
            if line == "END_0":
                pts.append(pts[0])
            else:
                splitted_line = line.split()
                point = [float(i) for i in splitted_line][:3]  # 只取点坐标，去掉法向量
                assert len(point) == 3, "点的坐标为x,y,z"
                pts.append(point)
        else:
            line = line.strip()
            splitted_line = line.split()
            point = [float(i) for i in splitted_line]
            assert len(point) == 3, "点的坐标为x,y,z"
            pts.append(point)

    f.close()
    pts = np.asarray(pts)
    return pts

path = r'E:\code\Test_Hgapi\AI_HTTP_gumline_simplify\UBP\12560-GM-W234 UBP.stl'
pts_path = r'E:\code\Test_Hgapi\AI_HTTP_gumline_simplify\UBP\12560-GM-W234 UBP.pts'
save_pts_path = os.path.splitext(pts_path)[0] + "_node.pts"
pts = get_gum_line_pts(pts_path)
with open(save_pts_path, "w") as f:
    f.write("BEGIN\n")
    for i in range(0, len(pts), 15):
        pt = pts[i]
        f.write(str(pt[0]) + " " + str(pt[1]) + " " + str(pt[2]) + "\n")
    f.write("END\n")

r = vtk.vtkSTLReader()
r.SetFileName(path)
r.Update()

arr = []
f1 = open(save_pts_path)

for i1 in f1.readlines()[1:-1]:

    b1 = i1.strip('\n')
    s1 = b1.split(' ')
    brr = []
    for j1 in s1:
        l1 = float(j1)
        brr.append(l1)
    arr.append(brr)
print(len(arr))


vtk_Points = vtk.vtkPoints()
vertices = vtk.vtkCellArray()

for v in arr:
    ids = vtk_Points.InsertNextPoint(v)
    vertices.InsertNextCell(1)
    vertices.InsertCellPoint(ids)

TemplateF = vtk.vtkPolyData()
TemplateF.SetPoints(vtk_Points)
TemplateF.SetVerts(vertices)
TemplateF.Modified()

ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(r.GetOutput())
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetOpacity(0.5)

contourRep = vtk.vtkOrientedGlyphContourRepresentation()
contourRep.GetLinesProperty().SetColor(0, 0, 1)
contourRep.GetLinesProperty().SetLineWidth(5)
contourRep.GetProperty().SetPointSize(5)

contourWidget = vtk.vtkContourWidget()
contourWidget.SetInteractor(iren)
contourWidget.SetRepresentation(contourRep)
contourWidget.On()
contourWidget.Initialize(TemplateF, 1)
contourWidget.CloseLoop()

pointPlacer = vtk.vtkPolygonalSurfacePointPlacer()
pointPlacer.AddProp(actor)
pointPlacer.GetPolys().AddItem(r.GetOutput())
pointPlacer.SnapToClosestPointOn()
contourRep.SetPointPlacer(pointPlacer)

ren.AddActor(actor)
ren.ResetCamera()
renWin.Render()

iren.Initialize()
iren.Start()

pro_model = load(path)

line_point = contourWidget.GetRepresentation().GetContourRepresentationAsPolyData()
points = vtk_to_numpy(line_point.GetPoints().GetData())

with open(path[0:-4] + '.pts', 'w') as fr:
    b = 'BEGIN_0' + '\n'
    e = 'END_0'

    fr.write(b)
    for i in points:
        ii = pro_model.closestPoint(i)
        for j in range(len(ii)):
            if j <= 1:
                fr.write(str(ii[j]) + ' ')

            else:
                # i[j] -= 0.1
                fr.write(str(ii[j]) + '\n')
    fr.write(e)
    fr.close()

    print('PTS File has saved....')
