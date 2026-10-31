# Trossen-Surge Leader Teleoperator

`WidowXAILeaderTeleopSurge` — wraps the plain `WidowXAILeaderTeleop` and a Manus
glove (`altbionics`) so `lerobot-teleoperate` / `lerobot-record` drive the arm
and the Surge hand together. Produces 6 arm + 6 hand `.pos` values (the gripper
is dropped; the hand replaces it).

Select with `--teleop.type=widowxai_leader_teleop_surge`.
