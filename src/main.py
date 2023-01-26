import typer
from detector.pose.pose_estimator import PoseEstimator
import cv2

app = typer.Typer()


@app.command()
def detect_pose(image_path: str):
    """
    Example: python3 src/main.py detect_pose ./assets/DropKnee.png

    Args:
        image_path: str
    """
    pose_estimator = PoseEstimator()

    while True:
        image = cv2.imread(f"{image_path}")

        pose = pose_estimator.get_pose(image)
        pose_estimator.draw_pose(image, pose)

        cv2.imshow("climbing", image)

        if cv2.waitKey(0 if image is not None else 1) == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    app()
