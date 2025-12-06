# Models

两个主要模型: Catmodel 和 Resnet

## **legacy 里是原来的代码**

## 主要修改：

1. lint 了文件，解决了所有警告

2. 合成了 predict_price.py 和 predict_pric+meta learning.py (原始文件备份在 legacy 里)

3. 所有的路径全部改为相对路径 都能直接运行跑通

4. 运行要求的库都在 requirements.txt 里 (用 `pipreqs . --force --encoding=utf8` 生成)
