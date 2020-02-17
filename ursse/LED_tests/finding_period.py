from numpy import mean,sqrt,var
from LED_tests.trig_times import find_trig_times
def find_period(ch2,trigger_level=0.5):
    trig_times = find_trig_times(ch2,trigger_level)
    def least_squares(X,Y):
        XY = [i*j for i,j in zip(X,Y)]
        X2 = [i**2 for i in X]
        meanX = mean(X)
        meanY=mean(Y)
        a = (mean(XY)-meanX*meanY)/(mean(X2)-meanX**2)
        b = meanY-a*meanX
        a_error = sqrt(var([a*i-j for i,j in zip(X,Y)])/sum([(x-meanX)**2 for x in X]))
        return [a,b,a_error]
    mean_period,offset,period_error = least_squares(range(len(trig_times)),trig_times)
    return (mean_period,period_error)



