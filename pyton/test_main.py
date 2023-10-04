from main import *
def test_area():
    assert Triangle(np.array([[0,0],[1,0],[0,1]])).area() == 0.5