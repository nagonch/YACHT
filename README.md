# YACHT: Yet Another Camera Hand calibration Tool

## Running the script
- ROS not required
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

- Edit `config.yaml`:
  - Set `data-folder` path
  - Set chessboard target size

- Prepare data:
  - `your_data_folder/images/` — images named like `0001.png` (order matters)
  - `your_data_folder/poses.npy` — [N, 4, 4] numpy array of poses
- Run:
  ```
  python yacht.py
  ```
- To load your result: `np.load("your_data_folder/result.npy")`

## Output & Checks
- Calibration RMS error should be low — otherwise rerecord the data
  
- Validate cam-to-arm pose and error — rerecord the data if high
  
- With `visualize-2D`:
  → your_data_folder/visualization/ contains calibration target frame reprojected into images

- With `visualize-3D`:
  - First view — camera poses w.r.t. target
  <img src="imgs/view1.png" width="500" />

  - Second view — camera, arm, target poses in global frame. Target frames (frames in front of cameras) should cluster tightly. If spread out — error is high, add more diverse data
  <img src="imgs/view2.png" width="500" />

## Example dataset
Located in `example_dataset` folder, provided by the US Department of Agriculture ([link](https://catalog.data.gov/dataset/data-from-solving-the-robot-world-hand-eyes-calibration-problem-with-iterative-methods-02035))

## Author's note
Made by [Nikolai Goncharov](https://www.linkedin.com/in/nikolai-goncharov-2931a31a5/). There's lots of "I've got a code for calibrating arm to camera, I can share if you want". Well, I just shared mine the way I did it.

<img src="imgs/yacht.jpg" width="500" />
