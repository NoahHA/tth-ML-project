# import pytest
# from src.models import train_model

# def test_typical():
#     assert train_model.train_RNN(1) == 1
#     assert train_model.train_RNN(2) == 1
#     assert train_model.train_RNN(6) == 8
#     assert train_model.train_RNN(40) == 102334155


# def test_edge_case():
#     assert train_model.train_RNN(0) == 0


# def test_raises():
#     with pytest.raises(NotImplementedError):
#         train_model.train_RNN(-1)

#     with pytest.raises(NotImplementedError):
#         train_model.train_RNN(1.5)

# TESTS:

# test that make_RNN_model returns a compiled model and that it only has one node in the last layer (binary classifier)
