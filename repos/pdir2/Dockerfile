# https://github.com/laike9m/pdir2 at the IDoFT commit. Build: docker build -t seqaccept/pdir2:1 repos/pdir2
FROM seqaccept/base:1
RUN git clone -q https://github.com/laike9m/pdir2 /repo && git -C /repo checkout -q ef5e2200074d68e2ca3df4ee441a1b9ea6ca5abc
WORKDIR /repo
RUN pip install --no-cache-dir  -e .
