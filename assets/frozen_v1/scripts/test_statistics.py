"""Synthetic numerical checks only; these arrays are not experimental data."""
import numpy as np
from summarize import make_intervals,rebin

def main():
    t=np.array([0,10,20,100,110,120],float)
    v=np.array([[1,3],[3,1],[5,0],[7,1],[9,3],[11,5]],float)
    valid=np.ones(6,bool)
    a=make_intervals(t,v,valid)
    assert len(a)==4 and sum(r['end']-r['start'] for r in a)==40
    assert a[2]['effective_start']==20 and a[2]['start']==100 and a[2]['segment']==1
    rows=rebin(a,15)
    assert np.isclose(sum(r['observed_seconds'] for r in rows),40)
    assert all(np.isclose(r['left_share']+r['right_share'],1) for r in rows)
    raw=rebin(a,30,'actual',120)
    assert raw[1]['coverage']==0 and np.isnan(raw[1]['left_mean'])
    assert np.isnan(raw[1]['left_insect_minutes'])
    valid[1]=False;b=make_intervals(t,v,valid)
    assert len(b)==2
    zero=make_intervals(np.array([0,15]),np.zeros((2,2)),np.ones(2,bool))
    assert np.isnan(rebin(zero,300)[0]['left_share'])
    repeated=make_intervals(np.array([0,0,10]),np.ones((3,2)),np.ones(3,bool))
    assert len(repeated)==1
    nan=make_intervals(np.array([0,10]),np.array([[1,2],[np.nan,2]]),np.ones(2,bool))
    assert not nan
    print('PASS: gaps, invalid frames, exact integration, boundary splitting, zero denominator, repeated timestamps, NaN counts')
if __name__=='__main__':main()
