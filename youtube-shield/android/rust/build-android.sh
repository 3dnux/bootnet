#!/usr/bin/env bash
# Compila libshieldtube.so (adblock-rust de Brave + puente JNI) para cada ABI de
# Android y la copia en app/src/main/jniLibs. Lo llama Gradle antes de compilar.
#
# Requiere: cargo, los targets de Rust para Android y el NDK
#   rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/../app/src/main/jniLibs}"
API=24

SDK="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}}"
NDK="${ANDROID_NDK_HOME:-$(ls -d "$SDK"/ndk/* 2>/dev/null | sort -V | tail -1)}"
if [[ ! -d "$NDK" ]]; then
  echo "No se encontró el NDK de Android (define ANDROID_NDK_HOME o instálalo con sdkmanager)" >&2
  exit 1
fi
case "$(uname -s)" in
  Linux) HOST=linux-x86_64 ;;
  Darwin) HOST=darwin-x86_64 ;;
  *) HOST=windows-x86_64 ;;
esac
BIN="$NDK/toolchains/llvm/prebuilt/$HOST/bin"

build() {
  local target="$1" abi="$2" clang="$3"
  local env_target="${target//-/_}"
  export "CC_${env_target}=$BIN/${clang}${API}-clang"
  export "AR_${env_target}=$BIN/llvm-ar"
  export "CARGO_TARGET_${env_target^^}_LINKER=$BIN/${clang}${API}-clang"
  cargo build --release --manifest-path "$HERE/Cargo.toml" --target "$target"
  mkdir -p "$OUT/$abi"
  cp "$HERE/target/$target/release/libshieldtube.so" "$OUT/$abi/"
}

build aarch64-linux-android arm64-v8a aarch64-linux-android
build armv7-linux-androideabi armeabi-v7a armv7a-linux-androideabi
build x86_64-linux-android x86_64 x86_64-linux-android
