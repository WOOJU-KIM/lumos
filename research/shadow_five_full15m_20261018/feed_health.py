"""Consumer-only latency accounting. Old frames are skipped, never made fresh."""
import math


class FeedHealth:
    def __init__(self, max_age):
        self.max_age = max_age
        self.degraded = False
        self.data_complete = True
        self.gaps = 0
        self.skipped = 0
        self.skipped_decisions = 0
        self.max_transport_ms = 0.
        self.max_queue_ms = 0.

    def gap(self):
        self.data_complete = False
        if self.degraded:
            return False
        self.degraded = True
        self.gaps += 1
        return True

    def admit(self, stamp, received, now, kind):
        self.max_transport_ms = max(self.max_transport_ms, (received-stamp)*1000)
        self.max_queue_ms = max(self.max_queue_ms, (now-received)*1000)
        age = now-stamp
        if not math.isfinite(age) or not 0 <= age <= self.max_age:
            self.skipped += 1
            self.skipped_decisions += int(kind == 2)
            return False
        return True

    def recovered(self):
        was_degraded = self.degraded
        self.degraded = False
        return was_degraded

    def snapshot(self):
        return dict(degraded=self.degraded, data_complete=self.data_complete,
                    gap_episodes=self.gaps, skipped_frames=self.skipped,
                    skipped_decisions=self.skipped_decisions,
                    max_transport_ms=self.max_transport_ms, max_queue_ms=self.max_queue_ms)
