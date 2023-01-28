import typer
from detector.pose.pose_estimator import PoseEstimator
from detector.holds.holds_detector import HoldsDetector
import cv2

app = typer.Typer()


@app.command()
def detect_pose(image_path: str):
    """
    Example: python3 src/main.py detect-pose ./assets/DropKnee.png

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
    Example: python3 src/main.py detect-holds ./assets/DropKnee.png

    Args:
        image_path: str
    """
    holds_detector = HoldsDetector()

    while True:
        image = cv2.imread(f"{image_path}")

        holds = holds_detector.vision_detect_holds(image, lower_area=100, upper_area=600)

        cv2.imshow("holds", holds)
        cv2.imshow("image", image)

        if cv2.waitKey(0 if image is not None else 1) == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    app()
