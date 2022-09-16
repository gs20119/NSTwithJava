

class Filter{
  Film[] kernels;
  int N;
  float bias;
  
  Filter(int n_, int size_, ArrayList<Float> params, float bias_){
    N = n_;
    kernels = new Film[N];
    bias = bias_;
    int nFilm = (2*size_+1)*(2*size_+1);
    for(int i=0; i<N; i++){
      ArrayList<Float> fparams = new ArrayList();
      for(int j=0; j<nFilm; j++) fparams.add(params.get(i*nFilm+j));
      kernels[i] = new Film(size_, fparams);
    }
  }
  ImageMap Conv(ImageMap[] x){
    ImageMap y = kernels[0].scan(x[0]);
    for(int i=1; i<N; i++){
      ImageMap temp = kernels[i].scan(x[i]);
      y.add(temp);
    }y.add(bias);
    return y;
  }
  ImageMap[] backConv(ImageMap dLdy_){
    ImageMap[] dLdx_ = new ImageMap[N];
    for(int i=0; i<N; i++){
      dLdx_[i] = kernels[i].backScan(dLdy_, i);
    }
    return dLdx_;
  }
}



class Film{
  int size;
  float[][] values;
  
  Film(int size_, ArrayList<Float> params){ // size 2N+1 * 2N+1
    size = size_;
    values = new float[2*size+1][2*size+1];
    for(int I=0; I<=2*size; I++)
      for(int J=0; J<=2*size; J++)
        values[I][J] = params.get(I*(2*size+1)+J);
  }
  
  float get(int a, int b){
    if(a<0 || 2*size<a || b<0 || 2*size<b) return 0; 
    return values[a][b];
  }
  
  ImageMap scan(ImageMap x){
    ImageMap y = new ImageMap(x.H, x.W);
    for(int i=0; i<y.H; i++)
      for(int j=0; j<y.W; j++){
        float sum = 0;
        for(int I=-size; I<=size; I++)
          for(int J=-size; J<=size; J++)
            sum += x.get(i+I,j+J)*this.get(size+I, size+J);
        y.set(i,j,sum);  
      }
    return y;
  }
  
  ImageMap backScan(ImageMap dLdy_, int trash){
    ImageMap dLdx__ = new ImageMap(dLdy_.H, dLdy_.W);
    for(int i=0; i<dLdx__.H; i++)
      for(int j=0; j<dLdx__.W; j++){
        float sum = 0.0;
        for(int I=-size; I<=size; I++)
          for(int J=-size; J<=size; J++){
            //if(i==2 && j==2 && I==size && trash==2) println(dLdy_.get(i+I,j+J) + " " + this.get(size-I, size-J));
            sum += dLdy_.get(i+I,j+J)*this.get(size-I, size-J);
          }
        dLdx__.set(i,j,sum); 
      }
    return dLdx__;
  }
  
}
