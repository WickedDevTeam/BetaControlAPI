{ pkgs }: {
  deps = [
    # Core dependencies
    pkgs.python39
    pkgs.nodejs-16_x
    pkgs.cmake
    pkgs.gcc

    # Python packages
    pkgs.python39Packages.pip
    pkgs.python39Packages.setuptools
    pkgs.python39Packages.wheel
    pkgs.python39Packages.numpy
    pkgs.python39Packages.flask
    pkgs.python39Packages.flask-cors
    pkgs.python39Packages.opencv4
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