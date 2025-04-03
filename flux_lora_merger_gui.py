# --- START OF FILE flux_lora_merger_gui_v6_final.py ---

import sys
import os
import logging
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QLineEdit, QMessageBox, QScrollArea, QGridLayout, QDoubleSpinBox,
    QSlider, QComboBox, QCheckBox
    # QApplication imported below for centering logic
)
from PyQt5.QtCore import QSettings, Qt
from safetensors.torch import load_file, save_file
import torch
from tqdm import tqdm # Optional, for progress bar

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

class LoraMerger(QWidget):
    # ... (ALL class methods: __init__, load_file, save_file, update_slider, etc. remain IDENTICAL to v5) ...
    # --- __init__ (Defaults: Strength=1.4, ClipRatio=0.0, Presets updated) ---
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flux LoRA Merger (Rank Padding)")
        self.settings = QSettings("BlackforestFluxTools", "LoraMergerRankPadding")
        self.last_dir = self.settings.value("last_dir", os.getcwd())
        logger.info(f"Loaded last directory: {self.last_dir}")
        self.lora1_label = QLabel("LoRA 1 (Base):"); self.lora1_input = QLineEdit(self.settings.value("lora1_path", "")); self.lora1_browse = QPushButton("Browse")
        self.lora2_label = QLabel("LoRA 2 (Detail):"); self.lora2_input = QLineEdit(self.settings.value("lora2_path", "")); self.lora2_browse = QPushButton("Browse")
        self.output_label = QLabel("Output LoRA File:"); self.output_input = QLineEdit(self.settings.value("output_path", "")); self.output_browse = QPushButton("Save As")
        self.unet_label = QLabel("Base UNet Model (Optional):"); self.unet_input = QLineEdit(self.settings.value("unet_path", "")); self.unet_browse = QPushButton("Browse")
        self.clip_label = QLabel("Base CLIP-L Model (Optional):"); self.clip_input = QLineEdit(self.settings.value("clip_path", "")); self.clip_browse = QPushButton("Browse")
        self.t5xxl_label = QLabel("Base T5-XXL Model (Optional):"); self.t5xxl_input = QLineEdit(self.settings.value("t5xxl_path", "")); self.t5xxl_browse = QPushButton("Browse")
        self.preset_label = QLabel("Merge Preset:"); self.preset_dropdown = QComboBox(); self.preset_dropdown.addItems(["", "All Blocks LoRA 1", "All Blocks LoRA 2", "All Double Blocks 1", "All Double Blocks 0", "All Single Blocks 1", "All Single Blocks 0"])
        self.strength_label = QLabel("Strength Multiplier (for Full Model Merge):"); self.strength_spinbox = QDoubleSpinBox(); self.strength_spinbox.setRange(0.0, 5.0); self.strength_spinbox.setSingleStep(0.1); self.strength_spinbox.setValue(float(self.settings.value("strength", 1.4))) # Default 1.4
        self.scale_lora_checkbox = QCheckBox("Scale Saved LoRA (Non-standard)"); self.scale_lora_checkbox.setToolTip("Applies strength multiplier directly to saved LoRA weights."); self.scale_lora_checkbox.setChecked(self.settings.value("scale_lora", False, type=bool))
        self.clip_ratio_label = QLabel("CLIP Merge Ratio (0.0 = LoRA1, 1.0 = LoRA2):"); self.clip_ratio_spinbox = QDoubleSpinBox(); self.clip_ratio_spinbox.setRange(0.0, 1.0); self.clip_ratio_spinbox.setSingleStep(0.05); self.clip_ratio_spinbox.setValue(float(self.settings.value("clip_ratio", 0.0))) # Default 0.0
        self.merge_full_checkbox = QCheckBox("Merge into Full Models (UNet, CLIP-L, T5-XXL)"); self.merge_full_checkbox.setChecked(self.settings.value("merge_full", False, type=bool))
        self.reset_button = QPushButton("Reset Sliders"); self.merge_button = QPushButton("Merge and Save")
        layout = QVBoxLayout()
        for label, input_field, button in [(self.lora1_label, self.lora1_input, self.lora1_browse), (self.lora2_label, self.lora2_input, self.lora2_browse), (self.output_label, self.output_input, self.output_browse), (self.unet_label, self.unet_input, self.unet_browse), (self.clip_label, self.clip_input, self.clip_browse), (self.t5xxl_label, self.t5xxl_input, self.t5xxl_browse)]: hbox = QHBoxLayout(); hbox.addWidget(label); hbox.addWidget(input_field); hbox.addWidget(button); layout.addLayout(hbox)
        preset_layout = QHBoxLayout(); preset_layout.addWidget(self.preset_label); preset_layout.addWidget(self.preset_dropdown); layout.addLayout(preset_layout)
        strength_layout = QHBoxLayout(); strength_layout.addWidget(self.strength_label); strength_layout.addWidget(self.strength_spinbox); layout.addLayout(strength_layout)
        scale_layout = QHBoxLayout(); scale_layout.addWidget(self.scale_lora_checkbox); layout.addLayout(scale_layout)
        clip_layout = QHBoxLayout(); clip_layout.addWidget(self.clip_ratio_label); clip_layout.addWidget(self.clip_ratio_spinbox); layout.addLayout(clip_layout)
        block_layout = QGridLayout(); block_layout.addWidget(QLabel("Block"), 0, 0); block_layout.addWidget(QLabel("Blend Ratio (0.0 = LoRA1, 1.0 = LoRA2)"), 0, 1); self.block_sliders = []
        row = 1; double_block_count = 19; single_block_count = 38
        logger.info(f"Initializing sliders for {double_block_count} double blocks and {single_block_count} single blocks.")
        for block_type, count in [("double_blocks", double_block_count), ("single_blocks", single_block_count)]:
            for i in range(count):
                block_name = f"{block_type}_{i}"; label = QLabel(block_name); slider = QSlider(Qt.Horizontal); slider.setRange(0, 100); slider.setValue(int(self.settings.value(f"slider_{block_name}", 50)))
                spinbox = QDoubleSpinBox(); spinbox.setRange(0.0, 1.0); spinbox.setSingleStep(0.01); spinbox.setValue(slider.value() / 100.0)
                slider.valueChanged.connect(lambda val, sp=spinbox, name=block_name: self.update_slider(sp, name, val)); spinbox.valueChanged.connect(lambda val, s=slider, name=block_name: self.update_spinbox(s, name, val))
                block_layout.addWidget(label, row, 0); block_layout.addWidget(slider, row, 1); block_layout.addWidget(spinbox, row, 2); self.block_sliders.append((block_name, slider, spinbox)); row += 1
        scroll = QScrollArea(); container = QWidget(); container.setLayout(block_layout); scroll.setWidget(container); scroll.setWidgetResizable(True); scroll.setMinimumHeight(300); layout.addWidget(scroll)
        button_layout = QHBoxLayout(); button_layout.addWidget(self.merge_full_checkbox); button_layout.addWidget(self.reset_button); button_layout.addWidget(self.merge_button); layout.addLayout(button_layout)
        self.setLayout(layout)
        self.lora1_browse.clicked.connect(lambda: self.load_file(self.lora1_input, "lora1_path")); self.lora2_browse.clicked.connect(lambda: self.load_file(self.lora2_input, "lora2_path"))
        self.output_browse.clicked.connect(lambda: self.save_file(self.output_input, "output_path")); self.unet_browse.clicked.connect(lambda: self.load_file(self.unet_input, "unet_path"))
        self.clip_browse.clicked.connect(lambda: self.load_file(self.clip_input, "clip_path")); self.t5xxl_browse.clicked.connect(lambda: self.load_file(self.t5xxl_input, "t5xxl_path"))
        self.merge_button.clicked.connect(self.merge_loras); self.reset_button.clicked.connect(self.reset_sliders); self.preset_dropdown.currentIndexChanged.connect(self.apply_preset)
        self.strength_spinbox.valueChanged.connect(lambda val: self.settings.setValue("strength", val)); self.scale_lora_checkbox.stateChanged.connect(lambda state: self.settings.setValue("scale_lora", bool(state)))
        self.clip_ratio_spinbox.valueChanged.connect(lambda val: self.settings.setValue("clip_ratio", val)); self.merge_full_checkbox.stateChanged.connect(lambda state: self.settings.setValue("merge_full", bool(state)))

    # --- load_file ---
    def load_file(self, line_edit, settings_key):
        file, _ = QFileDialog.getOpenFileName(self, "Select File", self.last_dir, "Safetensors files (*.safetensors)")
        if file: line_edit.setText(file); self.last_dir = os.path.dirname(file); self.settings.setValue("last_dir", self.last_dir); self.settings.setValue(settings_key, file); logger.info(f"Loaded '{settings_key}': {file}")

    # --- save_file ---
    def save_file(self, line_edit, settings_key):
        suggested_name = ""
        if not line_edit.text() and self.lora1_input.text() and self.lora2_input.text(): l1_base = os.path.splitext(os.path.basename(self.lora1_input.text()))[0]; l2_base = os.path.splitext(os.path.basename(self.lora2_input.text()))[0]; suggested_name = f"merged_padded_{l1_base}_{l2_base}.safetensors"
        default_path = os.path.join(self.last_dir, suggested_name)
        file, _ = QFileDialog.getSaveFileName(self, "Save Merged File", default_path, "Safetensors files (*.safetensors)")
        if file:
            if not file.endswith(".safetensors"): file += ".safetensors"
            line_edit.setText(file); self.last_dir = os.path.dirname(file); self.settings.setValue("last_dir", self.last_dir); self.settings.setValue(settings_key, file); logger.info(f"Set '{settings_key}': {file}")

    # --- update_slider ---
    def update_slider(self, spinbox, block_name, value): spinbox.setValue(value / 100.0); self.settings.setValue(f"slider_{block_name}", value)

    # --- update_spinbox ---
    def update_spinbox(self, slider, block_name, value): slider_value = int(value * 100); slider.setValue(slider_value) if slider.value() != slider_value else None; self.settings.setValue(f"slider_{block_name}", slider_value)

    # --- reset_sliders (Defaults: Strength=1.4, ClipRatio=0.0) ---
    def reset_sliders(self):
        logger.info("Resetting all settings to default."); default_slider_val = 50
        for _, slider, _ in self.block_sliders: slider.setValue(default_slider_val)
        self.clip_ratio_spinbox.setValue(0.0); self.strength_spinbox.setValue(1.4) # Defaults
        self.scale_lora_checkbox.setChecked(False); self.merge_full_checkbox.setChecked(False); self.preset_dropdown.setCurrentIndex(0)

    # --- apply_preset (Handles updated preset list) ---
    def apply_preset(self):
        preset = self.preset_dropdown.currentText(); target_val = 0
        if preset == "": return
        if "LoRA 2" in preset or "Blocks 1" in preset: target_val = 100
        logger.info(f"Applying preset: '{preset}' with target value {target_val}%")
        if "All Blocks" in preset: logger.info("Setting all block sliders."); [s.setValue(target_val) for _, s, _ in self.block_sliders]
        elif "All Double Blocks" in preset: logger.info("Setting double blocks to target, single blocks to 0."); [s.setValue(target_val if n.startswith("double") else 0) for n, s, _ in self.block_sliders]
        elif "All Single Blocks" in preset: logger.info("Setting single blocks to target, double blocks to 0."); [s.setValue(target_val if n.startswith("single") else 0) for n, s, _ in self.block_sliders]

    # --- get_block_from_lora_name ---
    def get_block_from_lora_name(self, lora_name):
        parts = lora_name.split('.'); relevant_part = parts[-1]
        for prefix in ["double_blocks_", "single_blocks_"]:
            if prefix in relevant_part: sub_parts = relevant_part.split(prefix); block_num_str = sub_parts[1].split('_')[0] if len(sub_parts) > 1 else None; return f"{prefix}{block_num_str}" if block_num_str and block_num_str.isdigit() else None
        return None

    # --- has_unet_weights ---
    def has_unet_weights(self, lora_sd): return any(k.startswith(("lora_unet_", "lora_flux_")) for k in lora_sd.keys())

    # --- has_clip_weights ---
    def has_clip_weights(self, lora_sd): return any(k.startswith(("lora_te_", "lora_clip_l_", "lora_t5_")) for k in lora_sd.keys())

    # --- apply_lora_to_base (Corrected Exception Handling) ---
    def apply_lora_to_base(self, lora_sd, base_sd, lora_name_to_base_key, strength_factor, working_device="cpu"):
        logger.info(f"Applying LoRA to base model on {working_device} with strength {strength_factor}")
        skipped_keys = []; applied_count = 0
        if not base_sd: logger.warning("Base state dict is empty during apply_lora_to_base."); return
        base_device = next(iter(base_sd.values())).device if base_sd else "cpu"
        for key in tqdm(list(lora_sd.keys()), desc="Applying LoRA to base"):
            if key.endswith(".lora_down.weight"):
                lora_name = key.replace(".lora_down.weight", ""); base_key = lora_name_to_base_key.get(lora_name)
                if base_key is None or base_key not in base_sd:
                    if lora_name not in skipped_keys: logger.warning(f"Base key not found for LoRA '{lora_name}'. Skipping apply."); skipped_keys.append(lora_name)
                    continue
                up_key = key.replace(".lora_down.weight", ".lora_up.weight"); alpha_key = lora_name + ".alpha"
                if up_key not in lora_sd: logger.warning(f"Missing up key '{up_key}' for '{lora_name}' during apply. Skipping."); skipped_keys.append(lora_name); continue
                try: # Try applying delta
                    down_weight = lora_sd[key].to(working_device, dtype=torch.float32); up_weight = lora_sd[up_key].to(working_device, dtype=torch.float32)
                    dim = down_weight.size(0); alpha = lora_sd.get(alpha_key, torch.tensor(dim)).item(); scale = alpha / dim if dim > 0 else 0
                    if base_key not in base_sd: logger.warning(f"Base key '{base_key}' missing before applying delta. Skipping."); skipped_keys.append(lora_name); continue
                    original_dtype = base_sd[base_key].dtype; weight = base_sd[base_key].to(working_device, dtype=torch.float32); delta = None
                    if len(weight.size()) == 2: delta = (up_weight @ down_weight) * scale * strength_factor
                    elif len(weight.size()) == 4 and down_weight.size()[2:4] == (1, 1): delta = (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * scale * strength_factor
                    elif len(weight.size()) == 4: delta = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3) * scale * strength_factor
                    else: logger.warning(f"Unsupported weight dim {len(weight.size())} for key {base_key} during apply."); skipped_keys.append(lora_name); continue
                    if delta is not None: weight += delta; base_sd[base_key] = weight.to(base_device, dtype=original_dtype); applied_count += 1
                    else: logger.warning(f"Delta calculation resulted in None for key {base_key}."); skipped_keys.append(lora_name)
                except Exception as e: # Catch errors during delta application
                    logger.error(f"Error applying LoRA delta to key {base_key}: {e}", exc_info=True); skipped_keys.append(lora_name)
                    try: # Attempt to restore original weight ON ERROR
                        if base_key in base_sd: base_sd[base_key] = base_sd[base_key].to(base_device, dtype=original_dtype)
                        else: logger.warning(f"Could not attempt restore for {base_key}: Key missing.")
                    except Exception as restore_e: logger.error(f"Failed to restore weight for key {base_key} after error: {restore_e}", exc_info=True)
        logger.info(f"Finished applying LoRA. Applied {applied_count} modules. Skipped/errored: {len(set(skipped_keys))} unique modules.")
        if skipped_keys: logger.warning(f"Unique skipped/errored LoRA modules during base apply: {list(set(skipped_keys))}")

    # --- Main Merge Logic (Includes Rank Padding) ---
    def merge_loras(self):
        lora1_path = self.lora1_input.text(); lora2_path = self.lora2_input.text(); output_path = self.output_input.text()
        if not (lora1_path and os.path.exists(lora1_path)): QMessageBox.warning(self, "Input Error", "Select valid LoRA 1."); return
        if not (lora2_path and os.path.exists(lora2_path)): QMessageBox.warning(self, "Input Error", "Select valid LoRA 2."); return
        if not output_path: self.save_file(self.output_input, "output_path"); output_path = self.output_input.text();
        if not output_path: QMessageBox.warning(self, "Input Error", "Specify output file path."); return

        unet_path = self.unet_input.text(); clip_path = self.clip_input.text(); t5xxl_path = self.t5xxl_input.text()
        merge_full = self.merge_full_checkbox.isChecked(); scale_lora = self.scale_lora_checkbox.isChecked()

        try:
            logger.info(f"Loading LoRA 1: {lora1_path}"); lora_sd1 = load_file(lora1_path, device="cpu")
            logger.info(f"Loading LoRA 2: {lora2_path}"); lora_sd2 = load_file(lora2_path, device="cpu")

            if merge_full: # Base model check
                 missing = []; has_unet = self.has_unet_weights(lora_sd1) or self.has_unet_weights(lora_sd2); has_clip = self.has_clip_weights(lora_sd1) or self.has_clip_weights(lora_sd2); has_t5 = any("lora_t5_" in k for k in set(lora_sd1.keys()).union(lora_sd2.keys()))
                 if has_unet and not (unet_path and os.path.exists(unet_path)): missing.append("UNet (Flux)")
                 if has_clip and not (clip_path and os.path.exists(clip_path)): missing.append("CLIP-L")
                 if has_t5 and not (t5xxl_path and os.path.exists(t5xxl_path)): missing.append("T5-XXL")
                 if missing: QMessageBox.warning(self, "Missing Base Models", f"Full merge checked, but missing:\n- {', '.join(missing)}"); return

            block_ratios = {name: spinbox.value() for name, _, spinbox in self.block_sliders}
            clip_ratio = self.clip_ratio_spinbox.value(); strength_factor = self.strength_spinbox.value()

            merged_data = {}; all_keys = set(lora_sd1.keys()).union(lora_sd2.keys())
            padded_count = 0; skipped_pairs = []

            logger.info("Starting LoRA merge process with rank padding...")
            for key in tqdm(all_keys, desc="Merging LoRA keys"):
                if key.endswith(".lora_down.weight"):
                    name = key.replace(".lora_down.weight", ""); up_key = key.replace(".lora_down.weight", ".lora_up.weight"); alpha_key = name + ".alpha"
                    pair1 = key in lora_sd1 and up_key in lora_sd1; pair2 = key in lora_sd2 and up_key in lora_sd2
                    if not (pair1 or pair2):
                        if name not in skipped_pairs: logger.warning(f"Skipping '{name}': Missing up/down pair."); skipped_pairs.append(name)
                        continue
                    block = self.get_block_from_lora_name(name); is_text = block is None; ratio = block_ratios.get(block, clip_ratio) if not is_text else clip_ratio
                    t1d = lora_sd1.get(key); t1u = lora_sd1.get(up_key); t2d = lora_sd2.get(key); t2u = lora_sd2.get(up_key); a1 = lora_sd1.get(alpha_key); a2 = lora_sd2.get(alpha_key)
                    eff_t1d, eff_t1u = t1d, t1u; eff_t2d, eff_t2u = t2d, t2u # Start with original tensors

                    if t1d is not None and t2d is not None: # Both LoRAs have this module, check rank
                        r1 = t1d.size(0); r2 = t2d.size(0)
                        if r1 != r2: # Ranks differ, apply padding
                            padded_count += 1; target = max(r1, r2)
                            if r1 < r2: # Pad 1 to match 2
                                eff_t1d = torch.zeros(target, t1d.size(1), dtype=t1d.dtype, device='cpu'); eff_t1d[:r1, :] = t1d
                                eff_t1u = torch.zeros(t1u.size(0), target, dtype=t1u.dtype, device='cpu'); eff_t1u[:, :r1] = t1u
                            else: # Pad 2 to match 1 (r2 < r1)
                                eff_t2d = torch.zeros(target, t2d.size(1), dtype=t2d.dtype, device='cpu'); eff_t2d[:r2, :] = t2d
                                eff_t2u = torch.zeros(t2u.size(0), target, dtype=t2u.dtype, device='cpu'); eff_t2u[:, :r2] = t2u

                    down_w, up_w, alpha_v = None, None, None # Calculate final weights
                    if eff_t1d is not None and eff_t2d is not None: # Average effective tensors
                        down_w = eff_t1d*(1-ratio) + eff_t2d*ratio; up_w = eff_t1u*(1-ratio) + eff_t2u*ratio
                        a1v = a1.item() if a1 is not None else t1d.size(0); a2v = a2.item() if a2 is not None else t2d.size(0); alpha_v = a1v*(1-ratio) + a2v*ratio
                    elif pair1: # Only LoRA 1 has pair
                        down_w = eff_t1d; up_w = eff_t1u; alpha_v = a1.item() if a1 is not None else eff_t1d.size(0)
                    elif pair2: # Only LoRA 2 has pair
                        down_w = eff_t2d; up_w = eff_t2u; alpha_v = a2.item() if a2 is not None else eff_t2d.size(0)
                    else: continue # Should not happen if pair check passed

                    if scale_lora: # Apply direct scaling if requested
                        if down_w is not None: down_w *= strength_factor
                        if up_w is not None: up_w *= strength_factor

                    # Store results
                    if down_w is not None: merged_data[key] = down_w
                    if up_w is not None: merged_data[up_key] = up_w
                    if alpha_v is not None: merged_data[alpha_key] = torch.tensor(alpha_v)

            if not merged_data: QMessageBox.critical(self, "Merge Error", "No mergeable keys found."); return # Final check
            logger.info(f"Merge complete. Padding applied to {padded_count} modules."); msg = f"Merged LoRA saved:\n{output_path}";
            if skipped_pairs: logger.warning(f"Skipped {len(skipped_pairs)} modules missing pairs.")
            if padded_count > 0: msg += f"\n\nNote: Rank padding applied to {padded_count} modules."
            logger.info(f"Saving merged LoRA ({len(merged_data)} tensors) to: {output_path}"); save_file(merged_data, output_path)

            if merge_full: # Merge into Full Models (Save to Original Dirs)
                logger.info("Merging into full models..."); parts = []; dev = "cuda" if torch.cuda.is_available() else "cpu"; logger.info(f"Using device: {dev}")
                m_str = strength_factor if not scale_lora else 1.0 # Use strength only if LoRA wasn't scaled
                for base_path, prefix_list, out_name_part, model_name in [(unet_path, ["lora_unet_", "lora_flux_"], "unet", "UNet"), (clip_path, ["lora_te_", "lora_clip_l_"], "clip-l", "CLIP-L"), (t5xxl_path, ["lora_t5_"], "t5xxl", "T5-XXL")]:
                    if base_path and os.path.exists(base_path):
                        try:
                            logger.info(f"Loading Base {model_name}: {base_path}"); base_sd = load_file(base_path, device="cpu")
                            key_map = {f"{pfx}{k.replace('.weight', '').replace('.', '_')}": k for k in base_sd if k.endswith(".weight") for pfx in prefix_list}
                            self.apply_lora_to_base(merged_data, base_sd, key_map, m_str, dev)
                            lora_n = os.path.splitext(os.path.basename(output_path))[0]; out_dir = os.path.dirname(base_path) # Original dir
                            out_path = os.path.join(out_dir, f"merged_{out_name_part}_{lora_n}.safetensors")
                            logger.info(f"Saving merged {model_name} to: {out_path}"); save_file(base_sd, out_path); parts.append(f"{model_name}: {out_path}"); del base_sd
                        except Exception as e: logger.error(f"Failed to merge {model_name}: {e}", exc_info=True); QMessageBox.warning(self, f"{model_name} Merge Error", f"Failed to merge {model_name}:\n{e}")
                if parts: msg += "\n\nFull models saved to original folders."

            logger.info("Saving final settings."); self.settings.setValue("lora1_path", lora1_path); self.settings.setValue("lora2_path", lora2_path); self.settings.setValue("output_path", output_path); self.settings.setValue("unet_path", unet_path); self.settings.setValue("clip_path", clip_path); self.settings.setValue("t5xxl_path", t5xxl_path); self.settings.setValue("strength", strength_factor); self.settings.setValue("scale_lora", scale_lora); self.settings.setValue("clip_ratio", clip_ratio); self.settings.setValue("merge_full", merge_full)
            QMessageBox.information(self, "Success", msg)

        except Exception as e: logger.error(f"Operation failed: {e}", exc_info=True); QMessageBox.critical(self, "Error", f"Failed:\n{e}\n\nCheck console log.")
        finally: logger.debug("Cleaning up tensors."); del lora_sd1, lora_sd2, merged_data; torch.cuda.empty_cache() if torch.cuda.is_available() else None

# --- Main execution block ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setOrganizationName("BlackforestFluxTools"); app.setApplicationName("LoraMergerRankPadding")

    window = LoraMerger()

    # --- Center and Resize Window ---
    desired_width = 800
    desired_height = 750 # Adjusted slightly
    window.resize(desired_width, desired_height)

    try:
        # Get available screen geometry (excludes taskbar, etc.)
        screen_geometry = app.primaryScreen().availableGeometry()
        screen_center = screen_geometry.center()

        # Get window frame geometry AFTER resizing
        frame_geometry = window.frameGeometry()
        frame_geometry.moveCenter(screen_center) # Move the frame's center

        # Move the window's top-left to the calculated frame top-left
        window.move(frame_geometry.topLeft())
        logger.info(f"Window resized to {desired_width}x{desired_height} and centered.")
    except Exception as center_e:
        logger.warning(f"Could not center or resize window: {center_e}")
    # --- End Center and Resize ---

    window.show()
    sys.exit(app.exec_())
# --- END OF FILE ---