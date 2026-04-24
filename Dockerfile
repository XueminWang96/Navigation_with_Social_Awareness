FROM python:3.13-slim

WORKDIR /app

COPY . /app

ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["python3", "social_navigation_demo.py", "--web"]
