{ pkgs }: {
  deps = [
    pkgs.python39
    pkgs.python39Packages.pip
    pkgs.python39Packages.flask
    pkgs.python39Packages.opencv4
    pkgs.python39Packages.numpy
    pkgs.python39Packages.pillow
    pkgs.python39Packages.dlib
    pkgs.python39Packages.requests
    pkgs.python39Packages.flask-cors
    pkgs.python39Packages.psutil
    pkgs.cmake
    pkgs.gcc
  ];
} 