BUILDDIR := build
SRCDIR := src

PEGDIR := peglib
PEGLINTDIR := $(PEGDIR)/lint
PEGLINTBUILDDIR := $(PEGLINTDIR)/build

src := $(SRCDIR)/compiler.cc
tgt := $(BUILDDIR)/compiler

run: $(tgt)
	$(tgt)

$(tgt): $(src) | $(BUILDDIR)
	clang++-13 -g -O1 -I$(PEGDIR) -pthread -rdynamic \
	`llvm-config-13 --cxxflags --ldflags --system-libs --libs core orcjit native` \
	-std=c++20 -fexceptions \
	$(src) -o $(tgt)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Lint
lint: $(PEGLINTBUILDDIR)/peglint
	xargs -0 -I {} ./$(PEGLINTBUILDDIR)/peglint $(SRCDIR)/grammar.peg --ast --source "{}"

$(PEGLINTBUILDDIR)/peglint: | $(PEGLINTBUILDDIR)
	cd $(PEGLINTBUILDDIR) && cmake .. -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_FLAGS="-pthread" && make

$(PEGLINTBUILDDIR):
	mkdir -p $(PEGLINTBUILDDIR)

.PHONY: run lint