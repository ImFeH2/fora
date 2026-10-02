use std::process::Command;

#[test]
fn prints_banner_and_exits() {
    let output = Command::new(env!("CARGO_BIN_EXE_fora-cli"))
        .output()
        .unwrap();

    assert!(output.status.success());
    assert_eq!(output.stdout, b"Fora CLI\n");
    assert!(output.stderr.is_empty());
}
