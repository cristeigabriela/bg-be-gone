"""The GTK4/libadwaita shell — the desktop half of bg-be-gone.

Everything in here is allowed to import `gi`, and nothing in here is allowed to
make a decision. The thinking (view transforms, hit-testing, animation, what to
draw) lives in `bgbg.engine`; this package translates GTK events into engine
events, applies the effects the engine hands back, and owns the pixels the engine
refers to only by handle.
"""
