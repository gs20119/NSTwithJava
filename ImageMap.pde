
class ImageMap{
  int W, H;
  float[][] values;
  
  ImageMap(int n_, int m_){
    H = n_; W = m_;
    values = new float[H][W];
  }
  void copy(ImageMap m){
    for(int i=0; i<H; i++)
      for(int j=0; j<W; j++)
        values[i][j] = m.get(i,j);
  }
  float get(int a, int b){
    if(a<0 || H<=a || b<0 || W<=b) return 0; // instead padding
    return values[a][b];
  }
  void set(int a, int b, float x){
    values[a][b] = x;
  }
  
  void add(ImageMap m){
    for(int i=0; i<H; i++)
      for(int j=0; j<W; j++)
        values[i][j] += m.get(i,j);
  }
  void sub(ImageMap m){
    for(int i=0; i<H; i++)
      for(int j=0; j<W; j++)
        values[i][j] -= m.get(i,j);
  }
  void add(float b){
    for(int i=0; i<H; i++)
      for(int j=0; j<W; j++)
        values[i][j] += b;
  }
  void mult(float b){
    for(int i=0; i<H; i++)
      for(int j=0; j<W; j++)
        values[i][j] *= b;
  }
  float product(ImageMap m){
    float sum = 0;
    for(int i=0; i<H; i++)
      for(int j=0; j<W; j++)
        sum += this.get(i,j)*m.get(i,j);
    return sum;
  }
}



class FixedImageMap extends ImageMap{
   Weight[][] weights;
   FixedImageMap(int n_, int m_){
     super(n_, m_);
     weights = new Weight[H][W];
     for(int i=0; i<H; i++)
       for(int j=0; j<W; j++)
         weights[i][j] = new Weight();
   }
   
   void optimize(ImageMap dLdx){
     for(int i=0; i<H; i++)
       for(int j=0; j<W; j++){
         weights[i][j].w = this.get(i,j);
         weights[i][j].grad = dLdx.get(i,j);
         adam.update(weights[i][j]);
         this.set(i,j,weights[i][j].w);
       }
   }
}
