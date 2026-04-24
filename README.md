# Social Navigation Demo

这是一个可交互的 `N-person social navigation` 演示项目。当前网页版本不是纯静态站点，它会持续请求服务端状态并发送交互命令，所以不适合直接部署到 `GitHub Pages`。

更适合的公开分享方式：

- 代码托管到 GitHub
- 用 `Render` 一键部署 Web Service
- 或者使用支持 `Dockerfile` 的平台，例如 `Railway`、`Fly.io`、`Cloud Run`

## 本地运行

```bash
python3 social_navigation_demo.py --web
```

默认访问地址：

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/dyad-triad`
- `http://127.0.0.1:8000/detour`

如果你想手动指定监听地址和端口：

```bash
python3 social_navigation_demo.py --web --host 0.0.0.0 --port 8000
```

也支持从环境变量读取：

```bash
HOST=0.0.0.0 PORT=8000 python3 social_navigation_demo.py --web
```

## 部署到 GitHub + Render

### 1. 初始化 Git 仓库

```bash
git init
git add .
git commit -m "Prepare social navigation demo for deployment"
```

### 2. 创建 GitHub 仓库并推送

把下面的 `<your-repo-url>` 替换成你的 GitHub 仓库地址：

```bash
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

### 3. 在 Render 上部署

1. 登录 Render
2. 选择 `New +` -> `Blueprint`
3. 连接你的 GitHub 仓库
4. Render 会读取仓库里的 `render.yaml`
5. 部署完成后，你会得到一个可公开访问的网址

健康检查地址：

```text
/healthz
```

主页面与两个子页面：

```text
/
/dyad-triad
/detour
```

## 其他托管方式

仓库中已经包含 `Dockerfile`，所以也可以部署到支持 Docker 的平台：

- Railway
- Fly.io
- Google Cloud Run
- 自己的云服务器

容器默认会监听：

- `HOST=0.0.0.0`
- `PORT=8000`

## 项目文件

- `social_navigation.py`: 社交导航核心逻辑
- `social_navigation_demo.py`: Tk demo + Web demo + HTTP 服务
- `render.yaml`: Render 部署配置
- `Dockerfile`: 通用容器部署配置
