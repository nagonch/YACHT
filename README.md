# YACHT: Yet Another Camera Hand calibration Tool

There's lots of "I've got a code for calibrating arm to camera, I can share if you want". Well, I'll just mine the way I did it.

![Yacht](yacht.jpg)

## Running the script
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

- Edit `config.yaml`:
  - Set `data-folder` path
  - Set chessboard target size

- Prepare data:
  - `your_data_folder/images/` — images named like `0001.png`
  - `your_data_folder/poses.npy` — [N, 4, 4] numpy array of poses
- Run:
  ```
  python yacht.py
  ```

## Output & Checks
- Calibration RMS error should be low — otherwise rerecord the data
  
- Validate cam-to-arm pose and error — rerecord the data if high
  
- With `visualize-2D`:
  → your_data_folder/visualization/ contains calibration target frame reprojected into images

- With `visualize-3D`:
  - First view — camera poses w.r.t. target

  - Second view — camera, arm, target poses in global frame. Target frames (frames in front of cameras) should cluster tightly. If spread out — error is high, add more diverse data

## TODOs
- [ ] Record test dataset
- [ ] Add demo video
- [ ] File standard for camera poses (ORDER MATTERS)
- [ ] File standard for calibration results
- [ ] Install instructions
- [ ] Data capture tips
- [ ] Try initial PnP guess
- [ ] Add comments for arguments
