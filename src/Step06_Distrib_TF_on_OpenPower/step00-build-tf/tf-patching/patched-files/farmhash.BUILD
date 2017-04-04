package(default_visibility = ["//visibility:public"])

prefix_dir = "farmhash-git"

genrule(
    name = "configure",
    srcs = glob(
        ["**/*"],
        exclude = [prefix_dir + "/config.h"],
    ),
    outs = [prefix_dir + "/config.h"],
    cmd = "pushd external/farmhash_archive/%s; workdir=$$(mktemp -d -t tmp.XXXXXXXXXX); cp -a * $$workdir; pushd $$workdir; ./configure; popd; popd; cp $$workdir/config.h $(@D); rm -rf $$workdir;" % prefix_dir,
)

cc_library(
    name = "farmhash",
    srcs = [prefix_dir + "/src/farmhash.cc"],
    hdrs = [prefix_dir + "/src/farmhash.h"] + [":configure"],
    includes = [prefix_dir],
    visibility = ["//visibility:public"]
)
