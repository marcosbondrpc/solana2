use std::io::Result;

fn main() -> Result<()> {
    // Compile protobuf schemas
    prost_build::Config::new()
        .bytes(&["."])
        .compile_protos(
            &["../../arbitrage-data-capture/protocol/realtime.proto"],
            &["../../arbitrage-data-capture/protocol/"],
        )?;
    
    // Set CPU optimization flags
    println!("cargo:rustc-link-arg=-Wl,-z,now");
    println!("cargo:rustc-link-arg=-Wl,-z,relro");
    println!("cargo:rustc-env=CARGO_CFG_TARGET_FEATURE=+avx2,+aes,+sse4.2");
    
    Ok(())
}