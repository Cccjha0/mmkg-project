# MMKG 后端草稿（Pydantic Schema + FastAPI Router）

这是一份可直接拷贝进 `backend/app/` 的后端草稿，目标是让后端同学按既定接口快速开工。

## 建议目录
```text
backend/
  app/
    main.py
    deps.py
    schemas/
    routers/
    services/
```

## 说明
- 这份草稿优先对齐当前前端 v3 需求，而不是做一个泛化后端。
- `Attribute Completion` 默认只服务 **Residual+Gate + OpenBG-IMG**。
- `Knowledge Graph` 默认只服务 **OpenBG-IMG**。
- `demo 商品清单` 由前端维护，因此这里没有提供 demo list 接口。
- `services/` 中目前放的是可运行的占位函数和 TODO 注释，便于后端逐步填实现。
