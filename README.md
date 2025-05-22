```docker build -t cifar-mlflow .```

```docker run -it --rm -p 4274:4274 -p 8880:8880 cifar-mlflow```

Запустить эксперимент с другими параметрами можно из bash внутри контейнера:

```docker run -it --rm -p 4274:4274 -p 8880:8880 cifar-mlflow bash```

MLFlow дашборд доступен по ```http://0.0.0.0:4272```

Jupyter Notebook поднят на ```http://127.0.0.1:8888/lab```
