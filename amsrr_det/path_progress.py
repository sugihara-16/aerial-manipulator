"""One-dimensional reference progress with discrete nominal stopping distance.

The fixed braking model is conditional, not a robust stopping certificate for
the articulated/contact system. Actual task execution remains independently
audited. An empty feasible intersection is a failure, never a fictitious hold.
"""
import numpy as np


def stopping_distance(speed,brake,period):
    """Hold -brake for whole periods, then one smaller no-reversal step."""
    speed=max(0.,float(speed))
    if brake<=0 or period<=0:raise ValueError('positive braking and period required')
    steps=int(np.floor(speed/(brake*period)))
    remainder=speed-steps*brake*period
    return steps*period*speed-.5*brake*(steps*period)**2+.5*period*remainder


class ScalarProgress:
    def __init__(self,distance,duration,period=.01):
        if distance<=0 or duration<=0:raise ValueError('positive path length and duration required')
        self.distance,self.period=distance,period
        self.peak=10/np.sqrt(3)*distance/duration**2
        self.maximum_speed=1.875*distance/duration
        self.position=0.;self.speed=0.;self.elapsed=0.;self.pause_position=None
        self.info=dict(scope=__doc__,nominal_brake_m_s2=self.peak,maximum_speed_m_s=self.maximum_speed,
                       steps=0,nominal_stopping_only=True)

    def pause(self):
        self.pause_position=min(self.distance,self.position+stopping_distance(self.speed,self.peak,self.period))

    def resume(self):self.pause_position=None

    @property
    def finished(self):return self.distance-self.position<1e-8 and self.speed<1e-8

    def choose(self,interval,*,observed_position=None,observed_speed=None,feedback_acceleration=0.,position_tolerance=.03):
        if interval is None:raise ValueError('no attainable path acceleration')
        h=self.period;end=self.distance if self.pause_position is None else self.pause_position
        lo=max(interval[0],-self.peak,-self.speed/h)
        hi=min(interval[1],self.peak,(self.maximum_speed-self.speed)/h)
        def excess(acceleration):
            speed=max(0.,self.speed+h*acceleration)
            position=self.position+h*self.speed+.5*h*h*acceleration
            violation=position+stopping_distance(speed,self.peak,h)-end
            if observed_position is not None:
                actual_acceleration=acceleration+feedback_acceleration
                actual_position=observed_position+h*observed_speed+.5*h*h*actual_acceleration
                actual_speed=max(0.,observed_speed+h*actual_acceleration)
                violation=max(violation,actual_position+stopping_distance(actual_speed,self.peak,h)-end-position_tolerance)
            return violation
        if hi<lo-1e-10 or excess(lo)>1e-9:raise ValueError('path cannot preserve nominal stopping domain')
        hi=max(lo,hi)
        if excess(hi)>0.:
            left,right=lo,hi
            for _ in range(45):
                mid=(left+right)/2
                if excess(mid)<=0.:left=mid
                else:right=mid
            hi=left
        return float(hi)

    def advance(self,acceleration):
        h=self.period;self.position+=h*self.speed+.5*h*h*acceleration
        self.speed=max(0.,self.speed+h*acceleration);self.elapsed+=h;self.info['steps']+=1
        self.info.update(position_m=self.position,speed_m_s=self.speed,elapsed_s=self.elapsed)

