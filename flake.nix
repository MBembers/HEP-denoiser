{
  description = "A Nix-flake-based C/C++ development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
  };

  outputs = { self, nixpkgs }: let
    lib = nixpkgs.lib;
    supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

    forSystem = system: let
      pkgs = import nixpkgs { inherit system; };
      pythonEnv = pkgs.python311.withPackages (ps: with ps; [ numpy scipy pandas matplotlib tabulate pybind11 ]);
      tools = with pkgs; [ 
        # clang
                #    clang-tools 
                # cmake codespell 
                # conan 
                # cppcheck 
                # doxygen 
                # vcpkg 
                # vcpkg-tool 
                # openblas 
                # lapack 
                # blas
                # cmake
                # fftw
            ];
      debugTools = if system != "aarch64-darwin" then [ pkgs.gdb ] else [];
    in pkgs.mkShell {
      buildInputs = tools ++ debugTools ++ [ pythonEnv pkgs.nix ];

      shellHook = ''
        echo "Entering dev shell for ${system}..."
      '';
    };
  in {
    devShells = lib.genAttrs supportedSystems (system: { default = forSystem system; });
  };
}
