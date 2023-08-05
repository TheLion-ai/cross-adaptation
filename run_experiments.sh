# poetry run python experiments/main.py -m classifier=xgb adapt_model=kliep
# poetry run python experiments/main.py -m classifier=xgb adapt_model=kmm
# poetry run python experiments/main.py -m classifier=xgb adapt_model=tradaboost

# poetry run python experiments/main.py -m classifier=dtree adapt_model=kliep
# poetry run python experiments/main.py -m classifier=dtree adapt_model=kmm

# poetry run python experiments/main.py -m classifier=knn adapt_model=kliep
# poetry run python experiments/main.py -m classifier=knn adapt_model=kmm

# poetry run python experiments/main.py -m classifier=mlp adapt_model=kliep
# poetry run python experiments/main.py -m classifier=mlp adapt_model=kmm

# poetry run python experiments/main.py -m classifier=nb adapt_model=kliep
# poetry run python experiments/main.py -m classifier=nb adapt_model=kmm

poetry run python experiments/main.py -m classifier=rf adapt_model=kliep
poetry run python experiments/main.py -m classifier=rf adapt_model=kmm

# poetry run python experiments/main.py -m classifier=svm adapt_model=kliep
# poetry run python experiments/main.py -m classifier=svm adapt_model=kmm

# poetry run python experiments/main.py -m classifier=dtree adapt_model=tradaboost
# poetry run python experiments/main.py -m classifier=knn adapt_model=tradaboost
# poetry run python experiments/main.py -m classifier=mlp adapt_model=tradaboost
# poetry run python experiments/main.py -m classifier=nb adapt_model=tradaboost
poetry run python experiments/main.py -m classifier=rf adapt_model=tradaboost
# poetry run python experiments/main.py -m classifier=svm adapt_model=tradaboost