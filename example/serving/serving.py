import matrixslow_serving as mss


serving = mss.serving.MatrixSlowServer('localhost:50051', './save/iris')
serving.serve()
