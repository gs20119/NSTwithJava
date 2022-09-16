
abstract class Optimizer{
}

class Weight{
  float w;
  float grad;
  float m, v;
  Weight(){
    m = 0; v = 0;
    grad = 0; w = 0;
  }
}

class Adam extends Optimizer{
  float b1 = 0.9, b2 = 0.999;
  float a = 10.0, eps = pow(10,-8);
  float b1n, b2n;
  Adam(){ b1n = 1; b2n = 1; }
  
  void timeplus(){ b1n *= b1; b2n *= b2; }
  void update(Weight W){
    W.m = b1*W.m + (1-b1)*W.grad;
    W.v = b2*W.v + (1-b2)*W.grad*W.grad;
    float mDecay = W.m/(1-b1n), vDecay = W.v/(1-b2n);
    //println(a*mDecay/(Math.sqrt(vDecay)+eps));
    W.w -= a*mDecay/(Math.sqrt(vDecay)+eps);
    W.grad = 0;
  }
}
