{ pkgs }: {
  deps = [
    # Core build tools
    pkgs.cmake
    pkgs.gcc
    pkgs.gnumake
    pkgs.pkg-config

    # Python and core packages
    pkgs.python39
    pkgs.python39Packages.pip
    pkgs.python39Packages.setuptools
    pkgs.python39Packages.wheel

    # Required system libraries
    pkgs.zlib
    pkgs.bzip2
    pkgs.openssl
    pkgs.libxml2
    pkgs.libxslt
    pkgs.libjpeg
    pkgs.openblas
    pkgs.boost

    # Image processing dependencies
    pkgs.opencv4
    pkgs.ffmpeg

    # Python packages that need compilation
    pkgs.python39Packages.numpy
    pkgs.python39Packages.pillow
    pkgs.python39Packages.flask
    pkgs.python39Packages.flask-cors
    pkgs.python39Packages.psutil
    pkgs.python39Packages.requests
  ];

  env = {
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.zlib
      pkgs.bzip2
      pkgs.openssl
      pkgs.libxml2
      pkgs.libxslt
      pkgs.libjpeg
      pkgs.openblas
      pkgs.boost
      pkgs.opencv4
    ];
    PYTHONPATH = "${pkgs.python39Packages.opencv4}/lib/python3.9/site-packages";
  };
} 