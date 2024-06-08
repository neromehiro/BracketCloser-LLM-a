# ベースイメージを指定
FROM nero1014/bracket-closer-image

# 一時ディレクトリのサイズを増やす
ENV TMPDIR=/tmp/docker

# 作業ディレクトリを設定
WORKDIR /app/project

# 必要なパッケージをインストール
RUN mkdir -p /tmp/docker && pip install optuna && pip install python-dotenv

# プロジェクトのソースコードをコピー
COPY . /app/project

# コンテナが起動したときに実行されるコマンドを指定
CMD ["bash"]
