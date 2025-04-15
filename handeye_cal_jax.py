import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as R
import jaxopt
from tqdm import tqdm


def get_poses_error(lhs_poses_T, rhs_poses_T, lambd=1.0):
    trans_error = (
        jnp.linalg.norm(lhs_poses_T[:, :3, 3] - rhs_poses_T[:, :3, 3], axis=1) ** 2
    )
    rot_error = (
        1
        - (
            R.from_matrix(lhs_poses_T[:, :3, :3]).as_quat()
            * R.from_matrix(rhs_poses_T[:, :3, :3]).as_quat()
        ).sum(axis=1)
        ** 2
    )
    return trans_error + lambd * rot_error


def get_loss(
    x,
    arm_to_base_rotation,
    arm_to_base_translation,
    target_to_cam_rotation,
    target_to_cam_translation,
):
    target_to_base_pose = x[:7]
    cam_to_arm_pose = x[7:]
    target_to_base_T = jnp.eye(4)
    target_to_base_T = target_to_base_T.at[:3, :3].set(
        R.from_quat(target_to_base_pose[3:]).as_matrix()
    )
    target_to_base_T = target_to_base_T.at[:3, 3].set(target_to_base_pose[:3])

    cam_to_arm_T = jnp.eye(4)
    cam_to_arm_T = cam_to_arm_T.at[:3, :3].set(
        R.from_quat(cam_to_arm_pose[3:]).as_matrix()
    )
    cam_to_arm_T = cam_to_arm_T.at[:3, 3].set(cam_to_arm_pose[:3])

    arm_to_base_T = jnp.stack(
        [
            jnp.eye(4),
        ]
        * arm_to_base_rotation.shape[0]
    )
    arm_to_base_T = arm_to_base_T.at[:, :3, :3].set(arm_to_base_rotation)
    arm_to_base_T = arm_to_base_T.at[:, :3, 3].set(arm_to_base_translation)

    target_to_cam_T = jnp.stack(
        [
            jnp.eye(4),
        ]
        * target_to_cam_rotation.shape[0]
    )
    target_to_cam_T = target_to_cam_T.at[:, :3, :3].set(target_to_cam_rotation)
    target_to_cam_T = target_to_cam_T.at[:, :3, -1].set(target_to_cam_translation)

    target_to_base_derived_T = arm_to_base_T @ cam_to_arm_T @ target_to_cam_T
    target_to_base_pred_T = jnp.stack(
        [
            target_to_base_T,
        ]
        * target_to_base_derived_T.shape[0]
    )
    return get_poses_error(target_to_base_derived_T, target_to_base_pred_T)


if __name__ == "__main__":
    arm_to_base_rotation = jnp.load("arm_to_base_rotation.npy")
    arm_to_base_translation = jnp.load("arm_to_base_translation.npy")
    target_to_cam_rotation = jnp.load("target_to_cam_rotation.npy")
    target_to_cam_translation = jnp.load("target_to_cam_translation.npy")
    target_to_base_pose = jnp.array([0, 0, 0, 0, 0, 0, 1], dtype=jnp.float32)
    cam_to_arm_pose = jnp.array([0, 0, 0, 0, 0, 0, 1], dtype=jnp.float32)
    x = jnp.concatenate((target_to_base_pose, cam_to_arm_pose))
    # print(
    #     get_loss(
    #         x,
    #         arm_to_base_rotation,
    #         arm_to_base_translation,
    #         target_to_cam_rotation,
    #         target_to_cam_translation,
    #     )
    # )
    optimizer = jaxopt.LevenbergMarquardt(
        get_loss,
        verbose=True,
    )
    state = optimizer.init_state(
        x,
        arm_to_base_rotation=arm_to_base_rotation,
        arm_to_base_translation=arm_to_base_translation,
        target_to_cam_rotation=target_to_cam_rotation,
        target_to_cam_translation=target_to_cam_translation,
    )
    params = x
    losses = []
    params_logs = []
    for i in tqdm(range(optimizer.maxiter)):
        opt_step = optimizer.update(
            params,
            state,
            arm_to_base_rotation=arm_to_base_rotation,
            arm_to_base_translation=arm_to_base_translation,
            target_to_cam_rotation=target_to_cam_rotation,
            target_to_cam_translation=target_to_cam_translation,
        )
        params = opt_step.params
        state = opt_step.state
        loss = state.error
        losses.append(loss)
        params_logs.append(params)

        if state.error < optimizer.tol:
            break

    losses = jnp.array(losses)
    params_logs = jnp.array(params_logs)
    params = params_logs[jnp.argmin(losses)]

    cam_to_arm_pose = params[7:]
    cam_to_arm_T = jnp.eye(4)
    cam_to_arm_T = cam_to_arm_T.at[:3, :3].set(
        R.from_quat(cam_to_arm_pose[3:]).as_matrix()
    )
    cam_to_arm_T = cam_to_arm_T.at[:3, 3].set(cam_to_arm_pose[:3])
    print(cam_to_arm_T)
