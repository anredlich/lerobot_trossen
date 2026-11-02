# Trossen-Surge Follower Robot

`WidowXAIFollowerSurge` — wraps the plain `WidowXAIFollower` and an Alt-Bionics
Surge hand (`altbionics`) so `lerobot-record` / `lerobot-teleoperate` drive the
arm and the hand together. State = 6 arm + 6 hand `.pos` (measured) + cameras;
action = 6 arm + 6 hand `.pos` (sent). The gripper is removed (pinned on the
command side, dropped from the dataset); the hand replaces it.

Select with `--robot.type=widowxai_follower_robot_surge`.
