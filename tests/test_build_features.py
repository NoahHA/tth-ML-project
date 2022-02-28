# import pytest
# from src.models import build_features

# # def test_typical():
# #     assert train_model.train_RNN(1) == 1
# #     assert train_model.train_RNN(2) == 1
# #     assert train_model.train_RNN(6) == 8
# #     assert train_model.train_RNN(40) == 102334155


# # def test_edge_case():
# #     assert train_model.train_RNN(0) == 0


# # def test_raises():
# #     with pytest.raises(NotImplementedError):
# #         train_model.train_RNN(-1)

# #     with pytest.raises(NotImplementedError):
# #         train_model.train_RNN(1.5)

# # TESTS:

# def test_loads_correct_data():
#     df = build_features.load_data(load_all=True)
#     assert df.shape == 