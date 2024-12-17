# transformer-chinese2english

## Base model

1. Run use cpu by default

2. Train model:

```bash
python nmt_model.py
```

The trained model is in the path: save/model.pt

3. Deployment:

```bash
python gradio_cn2en.py
```

Access the website provided by gradio.

## References

[基于Transformer的翻译模型（英-＞中）](https://blog.csdn.net/qq_44193969/article/details/116016404?spm=1001.2014.3001.5502)

[-transformer-english2chinese-](https://github.com/seanzhang-zhichen/-transformer-english2chinese-)