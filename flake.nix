{
  description = "Futhark MLIR backend";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    iree = {
      type = "git";
      url = "https://github.com/iree-org/iree.git";
      rev = "3699452d60c6dfe483df4675c7e98299fe790ec4";
      submodules = true;
      flake = false;
    };
  };

  outputs = { self, nixpkgs, iree }:
  let
    systems = [ "x86_64-linux" "aarch64-darwin" ];
    forAllSystems = f: nixpkgs.lib.genAttrs systems (system:
      f (import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      }));

    packageFor = pkgs: import ./nix/package.nix {
      inherit pkgs iree;
      src = ./.;
    };

    ireeToolsFor = pkgs: import ./nix/iree-tools.nix { inherit pkgs iree; };
  in {
    packages = forAllSystems (pkgs: {
      default = packageFor pkgs;
      iree-tools = ireeToolsFor pkgs;
    });

    devShells = forAllSystems (pkgs: {
      default = import ./nix/devshell.nix {
        inherit pkgs iree;
        mlir-backend = packageFor pkgs;
      };
    });
  };
}
