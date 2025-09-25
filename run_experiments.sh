# uv run python main.py -m classifier=rf adapt_model=kliep
# uv run python main.py -m classifier=rf adapt_model=kmm
# uv run python main.py -m classifier=rf adapt_model=tradaboost

uv run python main.py -m classifier=xgb adapt_model=kliep
# uv run python main.py -m classifier=xgb adapt_model=kmm


# uv run python main.py -m classifier=dtree adapt_model=kliep
# uv run python main.py -m classifier=dtree adapt_model=kmm
# uv run python main.py -m classifier=knn adapt_model=kliep

# uv run python main.py -m classifier=knn adapt_model=kmm
# uv run python main.py -m classifier=mlp adapt_model=kliep
# uv run python main.py -m classifier=mlp adapt_model=kmm

# uv run python main.py -m classifier=nb adapt_model=kliep
# uv run python main.py -m classifier=nb adapt_model=kmm

# uv run python main.py -m classifier=rf adapt_model=kliep
# uv run python main.py -m classifier=rf adapt_model=kmm

uv run python main.py -m classifier=svm adapt_model=kliep
# uv run python main.py -m classifier=svm adapt_model=kmm


# uv run python main.py -m classifier=dtree adapt_model=tradaboost
# uv run python main.py -m classifier=knn adapt_model=tradaboost
# uv run python main.py -m classifier=mlp adapt_model=tradaboost
# uv run python main.py -m classifier=nb adapt_model=tradaboost
# uv run python main.py -m classifier=rf adapt_model=tradaboost
# uv run python main.py -m classifier=svm adapt_model=tradaboost
# uv run python main.py -m classifier=xgb adapt_model=tradaboost

# uv run python main.py -m classifier=logistic adapt_model=tradaboost
# uv run python main.py -m classifier=logistic adapt_model=kliep
# uv run python main.py -m classifier=logistic adapt_model=kmm
