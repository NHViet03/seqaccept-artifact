# https://github.com/mtik00/yamicache at the IDoFT commit. Build: docker build -t seqaccept/yamicache:1 repos/yamicache
FROM seqaccept/base:1
RUN git clone -q https://github.com/mtik00/yamicache /repo && git -C /repo checkout -q 1109d08b18be94bdee55e113309f7be29f25d840
WORKDIR /repo
RUN pip install --no-cache-dir  -e .
