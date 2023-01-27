import typer
from detector.pose.pose_estimator import PoseEstimator
from detector.holds.holds_detector import HoldsDetector
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

        cv2.imshow("pose", image)

        if cv2.waitKey(0 if image is not None else 1) == ord('q'):
            cv2.destroyAllWindows()
            break


@app.command()
def detect_holds(image_path: str):
    """
    Example: python3 src/main.py detect_pose ./assets/DropKnee.png

    Args:
        image_path: str
    """
    holds_detector = HoldsDetector()

    while True:
        image = cv2.imread(f"{image_path}")

        holds = holds_detector.detect_holds(image)

        cv2.imshow("holds", holds)

        if cv2.waitKey(0 if image is not None else 1) == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    app()
