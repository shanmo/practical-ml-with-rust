#[cfg(test)]
mod assignment_generalised_distance {
    use ndarray::{Array, Array1, ArrayBase, Data, Ix1};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    /// To maximise the flexibility of this function's API, let's change the signature to make
    /// sure that we can call `euclidean_distance` using references to `Array`s, `ArrayView`s,
    /// `ArrayViewMut`s... or a combination of those!
    pub fn euclidean_distance(a: &ArrayBase<impl Data<Elem = f64>, Ix1>, b: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> f64
    {
        // You can probably copy-paste the body you just wrote in the previous exercise
        // If you are unlucky you might have to do some minor adjustments
        assert_eq!(a.len(), b.len());
        let mut result: f64 = 0.0; 
        for i in 0..a.len() {
            result += (a[i] - b[i]).powi(2);
        }
        result.sqrt()
    }

    #[test]
    // This test should compile.
    fn symmetry() {
        let n_features = 100;
        let mut a: Array1<f64> = Array::random(n_features, Uniform::new(-100., 100.));
        let mut b: Array1<f64> = Array::random(n_features, Uniform::new(-100., 100.));

        // euclidean_distance(&a, &b);
        // euclidean_distance(&a.view(), &b.view());
        euclidean_distance(&a.view_mut(), &b.view_mut());
        // euclidean_distance(&a, &b.view_mut());
        // euclidean_distance(&a.view(), &b.view_mut());
        // euclidean_distance(&a.view(), &b);
    }
}
