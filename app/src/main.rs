mod app;
mod classifier;
mod ui;

use app::MyApp;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "ML Visualizer",
        native_options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    )
}