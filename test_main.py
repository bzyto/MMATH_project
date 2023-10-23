from PoissonLinearApprox import *
def test_area():
    assert Triangle(np.array([[0,0],[1,0],[0,1]]), [0, 1, 2]).area() == 0.5
def test_UnitMeshHalf_NumberVertices():
    mesh = generateMesh_UnitSquare(0.5)
    assert len(mesh.vertices)==9
def test_UnitMeshHalf_NumberTriangles():
    mesh = generateMesh_UnitSquare(0.5)
    assert len(mesh.triangles) == 8
def test_ElementIntegrationMatrix():
    pass
def test_ElementIntegrationRHS():
    pass