use std::{marker, mem, ptr};

mod murmurhash3 {
    #[inline(always)]
    #[must_use]
    pub fn hash(key: u64) -> u64 {
        let mut t = key ^ (key >> 33);
        t = t.wrapping_mul(0xFF51AFD7ED558CCD);
        t ^= t >> 33;
        t = t.wrapping_mul(0xC4CEB9FE1A85EC53);
        t ^ (t >> 33)
    }
}

/// Compute the reminder of a for b assuming b is a power of 2.
#[inline(always)]
#[must_use]
fn mod_pow2_u64(a: u64, b: u64) -> u64 {
    debug_assert!(b.is_power_of_two());
    a & (b - 1)
}

/// Compute the reminder of a for b assuming b is a power of 2.
#[inline(always)]
#[must_use]
fn mod_pow2_usize(a: usize, b: usize) -> usize {
    debug_assert!(b.is_power_of_two());
    a & (b - 1)
}

/// Internal helper struct storing the information for managing the element in the map and the value.
#[repr(C)]
struct Element<T> {
    psl_and_key: u64,
    pub value: mem::MaybeUninit<T>,
}

/// Number of bits used for the PSL in the map.
const PSL_BITS: u32 = 7;
/// Number of bits available for the key.
pub const KEY_BITS: u32 = mem::size_of::<u64>() as u32 * 8 - PSL_BITS;
/// PSL of a free element, all the PSL_BITS set.
const FREE_ELEMENT_PSL: u64 = (1 << PSL_BITS) - 1;
/// Maximum PSL of an element before needing to rehash.
const MAX_PSL: u64 = FREE_ELEMENT_PSL - 1;
/// Maximum value of an element key.
pub const MAX_KEY: u64 = (1 << KEY_BITS) - 1;
/// Helper mask to extract the key.
const KEY_MASK: u64 = MAX_KEY;
/// Mask used to initialise the psl_and_key member.
const EMPTY_ELEMENT_MASK: u64 = !KEY_MASK;

impl<T> Element<T> {
    /// Helper function to merge the bits for a given psl and a key.
    #[inline(always)]
    #[must_use]
    fn merge_psl_and_key(psl: u64, key: u64) -> u64 {
        debug_assert!(psl <= MAX_PSL);
        debug_assert!(Self::is_valid_key(key));
        (psl << KEY_BITS) | key
    }

    /// Check if the key is valid.
    #[inline(always)]
    #[must_use]
    const fn is_valid_key(key: u64) -> bool {
        key <= MAX_KEY
    }

    /// Create a new Element for the given psl, key and value.
    #[inline(always)]
    #[must_use]
    fn new(psl: u64, key: u64, value: T) -> Self {
        Self {
            psl_and_key: Self::merge_psl_and_key(psl, key),
            value: mem::MaybeUninit::new(value),
        }
    }

    /// Extract the current psl of the element.
    #[inline(always)]
    #[must_use]
    const fn extract_psl(&self) -> u64 {
        self.psl_and_key >> KEY_BITS
    }

    /// Extract the key of the element.
    #[inline(always)]
    #[must_use]
    const fn extract_key(&self) -> u64 {
        self.psl_and_key & KEY_MASK
    }

    /// Check if the element is free. If not the value member MUST be initialised.
    #[inline(always)]
    #[must_use]
    const fn is_free(&self) -> bool {
        self.extract_psl() == FREE_ELEMENT_PSL
    }

    /// Unpack the Element into the key and the internal value.
    /// The element is reset to an empty one.
    #[inline(always)]
    #[must_use]
    unsafe fn unpack(&mut self) -> (u64, T) {
        debug_assert!(!self.is_free());
        // Extract the key of the element
        let key = self.extract_key();
        // Create new uninit value to insert
        let mut value = mem::MaybeUninit::uninit();
        // Swap the current value with the empty placeholder
        ptr::swap(self.value.as_mut_ptr(), value.as_mut_ptr());
        // Set the psl and key to the empty element mask
        self.psl_and_key = EMPTY_ELEMENT_MASK;
        // Return the key and extract the value from the swapped MaybeUninit
        (key, value.assume_init())
    }
}

impl<T> Default for Element<T> {
    #[inline]
    fn default() -> Self {
        Self {
            psl_and_key: EMPTY_ELEMENT_MASK,
            value: mem::MaybeUninit::uninit(),
        }
    }
}

impl<T> Drop for Element<T> {
    #[inline]
    fn drop(&mut self) {
        if !self.is_free() {
            unsafe { ptr::drop_in_place(self.value.as_mut_ptr()) };
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ConstructionError {
    InvalidBufferSize,
}

#[must_use]
pub enum TryInsertResult<T> {
    Success,
    ExistingKey,
    InvalidKey,
    OutOfSpace((u64, T)),
    MaxPslReached((u64, T)),
}

#[must_use]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum InsertResult {
    Success,
    ExistingKey,
    InvalidKey,
}

/// A very efficient and high performance hashmap based on robin-hood hashing.
/// Can work with arbitrary values but the key MUST be a u64 and only a maximum amount
/// of bits can be used
pub struct LinearHashMap<T> {
    // Flat array of element
    elements: Vec<Element<T>>,
    // Currently used elements
    in_use_elements: usize,
    // Maximum number of elements that can be in use
    max_in_use_elements: usize,
}

impl<T> LinearHashMap<T> {
    /// Fixed load factor for the map, this works well in general.
    const DEFAULT_LOAD_FACTOR: f64 = 0.875;
    /// Minimum valid size the map can have.
    const MIN_SIZE: usize = 8;
    /// Maximum valid size the map can have.
    const MAX_SIZE: usize = 1 << (8 * mem::size_of::<usize>() - 1);
    /// Maximum capacity the map can have.
    const MAX_CAPACITY: usize = (Self::MAX_SIZE as f64 * Self::DEFAULT_LOAD_FACTOR) as usize;

    /// Check if the given size is valid.
    #[must_use]
    fn is_valid_size(size: usize) -> bool {
        size.is_power_of_two() && (2..=Self::MAX_SIZE).contains(&size)
    }

    /// Create size for capacity.
    #[must_use]
    fn compute_size_for_capacity(capacity: usize) -> usize {
        debug_assert!(capacity <= Self::MAX_CAPACITY);
        let minimum_elements = (capacity as f64 / Self::DEFAULT_LOAD_FACTOR).ceil() as usize;
        minimum_elements.next_power_of_two().max(Self::MIN_SIZE)
    }

    /// Compute the next size for the map buffer. Returns None if we are at the maximum power of 2
    /// size for the usize type of the architecture.
    #[must_use]
    fn compute_next_size(&self) -> Option<usize> {
        let current_buffer_capacity = self.buffer_capacity();
        match current_buffer_capacity {
            0 => Some(Self::MIN_SIZE),
            Self::MAX_SIZE => None,
            _ => Some(2 * current_buffer_capacity),
        }
    }

    /// Helper function to create an empty buffer of elements.
    #[must_use]
    fn create_elements_buffer(size: usize) -> Vec<Element<T>> {
        debug_assert!(Self::is_valid_size(size));
        let mut buffer = Vec::with_capacity(size);
        for _ in 0..size {
            buffer.push(Element::default());
        }
        buffer
    }

    /// Helper function to check if a key is valid for the map or not.
    #[inline]
    #[must_use]
    pub const fn is_valid_key(key: u64) -> bool {
        Element::<T>::is_valid_key(key)
    }

    /// Create a new map that can hold at least the required capacity or more.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            Self::default()
        } else if capacity <= Self::MAX_CAPACITY {
            // Compute the number of elements required to satisfy the capacity and go to the next power of 2
            let size = Self::compute_size_for_capacity(capacity);
            let max_in_use_elements = Self::compute_max_in_use_element(size);
            debug_assert!(max_in_use_elements >= capacity);
            Self {
                elements: Self::create_elements_buffer(size),
                in_use_elements: 0,
                max_in_use_elements,
            }
        } else {
            panic!(
                "Required capacity for LinearHashMap {} is too large to satisfy",
                capacity
            );
        }
    }

    /// Create a new map with a buffer of the given size, returns an error if the size is not acceptable.
    pub fn with_buffer_size(size: usize) -> Result<Self, ConstructionError> {
        if Self::is_valid_size(size) {
            Ok(Self {
                elements: Self::create_elements_buffer(size),
                in_use_elements: 0,
                max_in_use_elements: Self::compute_max_in_use_element(size),
            })
        } else {
            Err(ConstructionError::InvalidBufferSize)
        }
    }

    /// Given a size, returns the maximum number of elements that can be used.
    #[inline(always)]
    #[must_use]
    fn compute_max_in_use_element(size: usize) -> usize {
        debug_assert!(Self::is_valid_size(size));
        (Self::DEFAULT_LOAD_FACTOR * size as f64) as usize
    }

    /// Given an index in the map, computes the next one by wrapping at the end.
    #[inline]
    #[must_use]
    fn compute_next_index(&self, index: usize) -> usize {
        debug_assert!((0..self.buffer_capacity()).contains(&index));
        mod_pow2_usize(index + 1, self.buffer_capacity())
    }

    /// For a given key, computes the index where the key would ideal go in the buffer.
    #[inline]
    #[must_use]
    fn compute_key_ideal_index(&self, key: u64) -> usize {
        debug_assert!(Self::is_valid_key(key));
        mod_pow2_u64(murmurhash3::hash(key), self.buffer_capacity() as u64) as usize
    }

    /// Returns the current number of elements in the map.
    #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        self.in_use_elements
    }

    /// Returns the current capacity of the buffer to store elements.
    #[inline]
    #[must_use]
    fn buffer_capacity(&self) -> usize {
        self.elements.len()
    }

    /// Returns the maximum number of elements that can be stored in the current buffer.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.max_in_use_elements
    }

    /// Resize the map by going to the next buffer size and try to insert all the keys already present.
    fn resize(&mut self) {
        // Check if we can increase the size
        let new_size = match self.compute_next_size() {
            Some(s) => s,
            None => panic!("Maximum size of the map has been reached"),
        };
        // Allocate new array of elements after removing the old ones
        let mut current_elements = Self::create_elements_buffer(new_size);
        mem::swap(&mut current_elements, &mut self.elements);
        // Set new values in the map
        self.in_use_elements = 0;
        self.max_in_use_elements = Self::compute_max_in_use_element(self.buffer_capacity());

        // Try to insert all old elements
        for e in &mut current_elements {
            if !e.is_free() {
                let (key, value) = unsafe { e.unpack() };
                match self.try_insert_internal(key, value, false) {
                    TryInsertResult::Success => (),
                    _ => panic!("Could not add key that was already in the map during resize"),
                }
            }
        }
    }

    /// Internal helper function to insert a new value in the map. The boolean value is there and
    /// always constant such that the internal check to see if the key is already there can be skipped.
    #[inline]
    fn try_insert_internal(
        &mut self,
        input_key: u64,
        input_value: T,
        check_existing_key: bool,
    ) -> TryInsertResult<T> {
        // Compute the index where the key should be
        let mut element_index = self.compute_key_ideal_index(input_key);
        // Create placeholder element to insert
        let mut to_insert_psl = 0;
        let mut to_insert_key = input_key;
        let mut to_insert_value = input_value;
        // Flag to check if the key is the input one
        let mut is_input_key = true;

        loop {
            // Get the candidate for insertion
            let insertion_candidate = unsafe { self.elements.get_unchecked_mut(element_index) };
            // Check if the element is free
            if insertion_candidate.is_free() {
                // Insert the element in the map
                insertion_candidate.psl_and_key =
                    Element::<T>::merge_psl_and_key(to_insert_psl, to_insert_key);
                unsafe { ptr::write(insertion_candidate.value.as_mut_ptr(), to_insert_value) };
                // Increase the number of elements in use and return success
                self.in_use_elements += 1;
                return TryInsertResult::Success;
            }
            // If requested, check if the input key matches the one we want to insert
            if check_existing_key
                && is_input_key
                && to_insert_key == insertion_candidate.extract_key()
            {
                return TryInsertResult::ExistingKey;
            }
            // Check if the PSL of the element to insert is larger than the one of the candidate, swap in that case
            if to_insert_psl > insertion_candidate.extract_psl() {
                // Set that we are not processing the input key anymore
                is_input_key = false;
                // Store temporary the psl and the key to insert
                let t_psl = to_insert_psl;
                let t_key = to_insert_key;
                // Store the value of the candidate in the local variables for the next steps
                to_insert_psl = insertion_candidate.extract_psl();
                to_insert_key = insertion_candidate.extract_key();
                // Set the new values in the candidate
                insertion_candidate.psl_and_key = Element::<T>::merge_psl_and_key(t_psl, t_key);
                // Swap the values
                unsafe { ptr::swap(&mut to_insert_value, insertion_candidate.value.as_mut_ptr()) };
            }

            // Increment the PSL for the next step
            to_insert_psl += 1;

            // Check if we are out of psl
            if to_insert_psl > MAX_PSL {
                return TryInsertResult::MaxPslReached((to_insert_key, to_insert_value));
            }

            // Go to the next index for the candidate element
            element_index = self.compute_next_index(element_index);
        }
    }

    /// Try to insert an element, this methods does not resize if it's not possible to insert it.
    pub fn try_insert(&mut self, input_key: u64, input_value: T) -> TryInsertResult<T> {
        if Self::is_valid_key(input_key) {
            // Check if we are out of space
            if self.size() == self.max_in_use_elements {
                TryInsertResult::OutOfSpace((input_key, input_value))
            } else {
                self.try_insert_internal(input_key, input_value, true)
            }
        } else {
            TryInsertResult::InvalidKey
        }
    }

    /// Insert an element in the map, resizes the map if the map is out of space or we did reach
    /// the maximum psl.
    pub fn insert(&mut self, input_key: u64, input_value: T) -> InsertResult {
        let mut to_insert_key = input_key;
        let mut to_insert_value = input_value;
        loop {
            match self.try_insert(to_insert_key, to_insert_value) {
                TryInsertResult::Success => return InsertResult::Success,
                TryInsertResult::ExistingKey => return InsertResult::ExistingKey,
                TryInsertResult::InvalidKey => return InsertResult::InvalidKey,
                TryInsertResult::OutOfSpace(element) | TryInsertResult::MaxPslReached(element) => {
                    self.resize();
                    to_insert_key = element.0;
                    to_insert_value = element.1;
                }
            }
        }
    }

    /// Helper function to find the index of given key. Returns None if the key is not found.
    #[inline]
    #[must_use]
    fn find_key_index(&self, key: u64) -> Option<usize> {
        // This check should be removed directly at compile time since MAX_PSL is a constant
        assert_eq!(MAX_PSL % 2, 0);
        const NUM_PASSES: usize = MAX_PSL as usize / 2;

        // Check if the key is valid
        if Self::is_valid_key(key) {
            let mut element_index = self.compute_key_ideal_index(key);
            let mut current_psl = 0;
            for _ in 0..=NUM_PASSES {
                // Get element from current index
                let element = unsafe { self.elements.get_unchecked(element_index) };
                // Check stop condition, either the element is free or the element psl is smaller
                // that what the key would have in this slot
                debug_assert!(current_psl <= MAX_PSL);
                if element.is_free() || element.extract_psl() < current_psl {
                    return None;
                }
                // Check if the key match
                if element.extract_key() == key {
                    return Some(element_index);
                }
                // Go to the next element index
                current_psl += 1;
                element_index = self.compute_next_index(element_index);

                // This is basically the same as the above
                // Unrolling it manually gives quite a big performance gain
                let element = unsafe { self.elements.get_unchecked(element_index) };
                debug_assert!(current_psl <= MAX_PSL);
                if element.is_free() || element.extract_psl() < current_psl {
                    return None;
                }
                if element.extract_key() == key {
                    return Some(element_index);
                }
                current_psl += 1;
                element_index = self.compute_next_index(element_index);
            }
        }
        None
    }

    /// Searches the map for the given key and returns a reference to the value.
    /// Returns None if the key is not there.
    #[inline]
    #[must_use]
    pub fn find(&self, key: u64) -> Option<&T> {
        self.find_key_index(key)
            .map(|index| unsafe { &(*self.elements.get_unchecked(index).value.as_ptr()) })
    }

    /// Searches the map for the given key and returns a mutable reference to the value.
    /// Returns None if the key is not there.
    #[inline]
    #[must_use]
    pub fn find_mut(&mut self, key: u64) -> Option<&mut T> {
        self.find_key_index(key).map(|index| unsafe {
            &mut (*self.elements.get_unchecked_mut(index).value.as_mut_ptr())
        })
    }

    /// Checks if a key is present in the map
    #[inline]
    #[must_use]
    pub fn has_key(&self, key: u64) -> bool {
        self.find_key_index(key).is_some()
    }

    /// Try to remove an element by the given key from the map and returns it.
    /// Returns None if the key is not there.
    pub fn remove(&mut self, key: u64) -> Option<T> {
        // First we try to see if the key is there or not
        match self.find_key_index(key) {
            Some(mut i) => {
                // Unpack the element to remove such that we can return the value at the end
                let (_, value) = unsafe { self.elements.get_unchecked_mut(i).unpack() };

                // Now we keep iterating until we find an element with either PSL equal to 0 or
                // an empty slot. While doing this, we shift back the elements and decrement their
                loop {
                    let next_element_index = self.compute_next_index(i);
                    let next_element =
                        unsafe { self.elements.get_unchecked_mut(next_element_index) };
                    if next_element.is_free() || next_element.extract_psl() == 0 {
                        // We are done, just return the taken out value
                        return Some(value);
                    } else {
                        // We insert the element in the previous slot and clear the next
                        unsafe {
                            // Extract data to insert
                            let next_element_psl = next_element.extract_psl();
                            debug_assert!(next_element_psl >= 1);
                            let (next_element_key, next_element_value) = next_element.unpack();
                            // Insert data in previous element
                            let current_element = self.elements.get_unchecked_mut(i);
                            debug_assert!(current_element.is_free());
                            *current_element = Element::new(
                                next_element_psl - 1,
                                next_element_key,
                                next_element_value,
                            );
                        }
                        // Update i to be the index of the next element
                        i = next_element_index;
                    }
                }
            }
            None => None,
        }
    }

    #[inline]
    #[must_use]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(self)
    }

    #[inline]
    #[must_use]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut::new(self)
    }

    #[inline]
    #[must_use]
    pub fn keys(&self) -> Keys<'_, T> {
        Keys { iter: self.iter() }
    }

    #[inline]
    #[must_use]
    pub fn values(&self) -> Values<'_, T> {
        Values { iter: self.iter() }
    }

    #[inline]
    #[must_use]
    pub fn values_mut(&mut self) -> ValuesMut<'_, T> {
        ValuesMut {
            iter: self.iter_mut(),
        }
    }
}

impl<T> Default for LinearHashMap<T> {
    #[inline]
    fn default() -> Self {
        Self {
            elements: Vec::new(),
            in_use_elements: 0,
            max_in_use_elements: 0,
        }
    }
}

impl<T: std::fmt::Display> std::fmt::Display for LinearHashMap<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for i in 0..self.buffer_capacity() {
            let e = unsafe { self.elements.get_unchecked(i) };
            if e.is_free() {
                writeln!(f, "Array element: {}, FREE", i)?;
            } else {
                writeln!(
                    f,
                    "Array element: {}, Key: {}, PSL: {}, Value: {}",
                    i,
                    e.extract_key(),
                    e.extract_psl(),
                    unsafe { &(*e.value.as_ptr()) },
                )?;
            }
        }
        Ok(())
    }
}

/// Helper macro to generate the body of next for iterators.
macro_rules! iterator_next_body {
    ($self:ident) => {
        if $self.is_done() {
            None
        } else {
            $self.advance_to_next_non_free_element();
            if $self.is_done() {
                return None;
            }
            let r = $self.get_key_value_tuple();
            // Go to the next step such that next time we start to check from the next element
            $self.go_to_next_element();
            Some(r)
        }
    };
}

/// Iterator over the couples (key, value)
pub struct Iter<'a, T> {
    current_element: *const Element<T>,
    last_element: *const Element<T>,
    _marker: marker::PhantomData<&'a Element<T>>,
}

impl<'a, T> Iter<'a, T> {
    #[must_use]
    fn new(map: &'a LinearHashMap<T>) -> Self {
        let first_element = map.elements.as_ptr();
        Self {
            current_element: first_element,
            last_element: unsafe { first_element.add(map.buffer_capacity()) },
            _marker: marker::PhantomData,
        }
    }

    #[inline]
    fn advance_to_next_non_free_element(&mut self) {
        debug_assert!(self.current_element < self.last_element);
        unsafe {
            while (*self.current_element).is_free() && !self.is_done() {
                self.go_to_next_element();
            }
        }
    }

    #[inline]
    fn go_to_next_element(&mut self) {
        debug_assert!(self.current_element < self.last_element);
        self.current_element = unsafe { self.current_element.add(1) };
    }

    #[inline]
    #[must_use]
    fn get_key_value_tuple(&self) -> (u64, &'a T) {
        unsafe {
            debug_assert!(!(*self.current_element).is_free());
            let e = &(*self.current_element);
            (e.extract_key(), &(*e.value.as_ptr()))
        }
    }

    #[inline]
    #[must_use]
    fn is_done(&self) -> bool {
        self.current_element >= self.last_element
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (u64, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        iterator_next_body!(self)
    }
}

/// Iterator over the keys
pub struct Keys<'a, T> {
    iter: Iter<'a, T>,
}

impl<'a, T> Iterator for Keys<'a, T> {
    type Item = u64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(key, _)| key)
    }
}

/// Iterator over the values
pub struct Values<'a, T> {
    iter: Iter<'a, T>,
}

impl<'a, T> Iterator for Values<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }
}

/// Mutable iterator over the couples (key, mut value)
pub struct IterMut<'a, T> {
    current_element: *mut Element<T>,
    last_element: *mut Element<T>,
    _marker: marker::PhantomData<&'a mut Element<T>>,
}

impl<'a, T> IterMut<'a, T> {
    #[must_use]
    fn new(map: &'a mut LinearHashMap<T>) -> Self {
        let first_element = map.elements.as_mut_ptr();
        Self {
            current_element: first_element,
            last_element: unsafe { first_element.add(map.buffer_capacity()) },
            _marker: marker::PhantomData,
        }
    }

    #[inline]
    fn advance_to_next_non_free_element(&mut self) {
        debug_assert!(self.current_element < self.last_element);
        unsafe {
            while (*self.current_element).is_free() || self.is_done() {
                self.go_to_next_element();
            }
        }
    }

    #[inline]
    fn go_to_next_element(&mut self) {
        debug_assert!(self.current_element != self.last_element);
        self.current_element = unsafe { self.current_element.add(1) };
    }

    #[inline]
    #[must_use]
    fn get_key_value_tuple(&mut self) -> (u64, &'a mut T) {
        unsafe {
            debug_assert!(!(*self.current_element).is_free());
            let e = &mut (*self.current_element);
            (e.extract_key(), &mut (*e.value.as_mut_ptr()))
        }
    }

    #[inline]
    #[must_use]
    fn is_done(&self) -> bool {
        self.current_element >= self.last_element
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = (u64, &'a mut T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        iterator_next_body!(self)
    }
}

/// Iterator over the values
pub struct ValuesMut<'a, T> {
    iter: IterMut<'a, T>,
}

impl<'a, T> Iterator for ValuesMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    type TestHashMap = LinearHashMap<i32>;

    #[test]
    fn test_creation() {
        // Test capacity
        const REQUIRED_CAPACITY: usize = 7;
        let m = TestHashMap::with_capacity(REQUIRED_CAPACITY);
        assert_eq!(m.size(), 0);
        assert!(m.capacity() >= REQUIRED_CAPACITY);
        assert_eq!(m.buffer_capacity(), 8);
        // Test buffer size
        for i in 1..15 {
            let size = 1 << i;
            match TestHashMap::with_buffer_size(size) {
                Ok(_) => (),
                Err(e) => {
                    assert_eq!(e, ConstructionError::InvalidBufferSize);
                    panic!("Could not construct map with size that should be valid");
                }
            }
        }
        match TestHashMap::with_buffer_size(0) {
            Ok(_) => panic!("Created map that should not be possible to create"),
            Err(e) => assert_eq!(e, ConstructionError::InvalidBufferSize),
        }
        match TestHashMap::with_buffer_size(1) {
            Ok(_) => panic!("Created map that should not be possible to create"),
            Err(e) => assert_eq!(e, ConstructionError::InvalidBufferSize),
        }
        match TestHashMap::with_buffer_size(10) {
            Ok(_) => panic!("Created map that should not be possible to create"),
            Err(e) => assert_eq!(e, ConstructionError::InvalidBufferSize),
        }
    }

    #[test]
    fn test_try_insert() {
        let mut map = match TestHashMap::with_buffer_size(8) {
            Ok(m) => m,
            Err(_) => panic!("Map construction failed in try_insert test"),
        };
        // Try to insert some values
        for i in 0..map.capacity() {
            match map.try_insert(i as u64, i as i32) {
                TryInsertResult::Success => (),
                _ => panic!("Simple value insertion failed"),
            }
        }
        // Try to insert a value that should return we are out of space
        const OUT_OF_SPACE_KEY: u64 = 123_123;
        const OUT_OF_SPACE_VALUE: i32 = 100_000;
        match map.try_insert(OUT_OF_SPACE_KEY, OUT_OF_SPACE_VALUE) {
            TryInsertResult::OutOfSpace(p) => {
                assert_eq!(p.0, OUT_OF_SPACE_KEY);
                assert_eq!(p.1, OUT_OF_SPACE_VALUE);
            }
            _ => panic!("Insert did not fail as expected"),
        }
    }

    #[test]
    fn test_invalid_key_insert() {
        let mut map = TestHashMap::default();
        const INVALID_KEY: u64 = MAX_KEY + 1;
        match map.insert(INVALID_KEY, 0) {
            InsertResult::InvalidKey => (),
            _ => panic!("Adding invalid key did not fail as expected"),
        }
    }

    #[test]
    fn test_insert() {
        let mut map = TestHashMap::default();

        // Try to insert some values
        const START_KEY: u64 = 0;
        const END_KEY: u64 = 100_000;
        for key in START_KEY..=END_KEY {
            match map.insert(key, key as i32) {
                InsertResult::Success => (),
                _ => panic!("Could not insert valid key"),
            }
        }

        // Try to insert the same values
        for key in START_KEY..=END_KEY {
            match map.insert(key, 0) {
                InsertResult::ExistingKey => (),
                _ => panic!("Existing key was not found correctly"),
            }
        }
    }

    #[test]
    fn test_lookup() {
        let mut map = TestHashMap::default();
        const START_KEY: u64 = 0;
        const END_KEY: u64 = 100_000;
        for key in START_KEY..=END_KEY {
            match map.insert(key, key as i32) {
                InsertResult::Success => (),
                _ => panic!("Failed to add a key when we should be able to"),
            }
        }

        // Lookup the values we expect to find
        for key in START_KEY..=END_KEY {
            match map.find(key) {
                Some(&v) => assert_eq!(v, key as i32),
                None => panic!("Could not find key that should be in the map"),
            }
        }

        // Lookup some values that should not be in the map
        const MISSING_KEY_OFFSET: u64 = 1000;
        for key in END_KEY + 1..=END_KEY + MISSING_KEY_OFFSET {
            if map.find(key).is_some() {
                panic!("Found key that should not be in the map");
            }
        }
    }

    #[test]
    fn test_remove() {
        let mut map = TestHashMap::default();
        const START_KEY: u64 = 0;
        const END_KEY: u64 = 1000;
        for key in START_KEY..=END_KEY {
            match map.insert(key, key as i32) {
                InsertResult::Success => (),
                _ => panic!("Failed to add a key when we should be able to"),
            }
        }

        // Try to remove some values that are not in the map
        const MISSING_TO_REMOVE_OFFSET: u64 = 1000;
        for key in END_KEY + 1..=END_KEY + MISSING_TO_REMOVE_OFFSET {
            if map.remove(key).is_some() {
                panic!("Removed key that should not be in the map");
            }
        }

        // Now for each value, we remove it and make sure all other values are still there and the one we removed is gone
        for key in (START_KEY..=END_KEY).rev() {
            match map.remove(key) {
                Some(v) => {
                    assert_eq!(v, key as i32);
                }
                None => panic!("Could not remove key that was expected to be removed successfully"),
            }
            // Now make sure that all other keys are still in there and can be found
            for test_key in START_KEY..key {
                match map.find(test_key) {
                    Some(&v) => assert_eq!(v, test_key as i32),
                    None => panic!("Cloud not find key that should still be there after removal"),
                }
            }
        }
    }

    #[test]
    fn test_iterator() {
        let mut map = TestHashMap::default();
        const START_KEY: u64 = 0;
        const END_KEY: u64 = 1000;
        for key in START_KEY..END_KEY {
            match map.insert(key, key as i32) {
                InsertResult::Success => (),
                _ => panic!("Failed to add a key when we should be able to"),
            }
        }
        // Test iterator over keys and values
        let mut found_buffer = vec![false; END_KEY as usize];
        for (k, v) in map.iter() {
            // Check the key is something we expected
            assert!((START_KEY..END_KEY).contains(&k));
            if found_buffer[k as usize] {
                panic!("Found key two times");
            } else {
                found_buffer[k as usize] = true;
                assert_eq!(k as i32, *v);
            }
        }
        // Check all the keys where found
        assert!(found_buffer.iter().all(|b| *b));

        // Test iterator over keys
        let mut found_buffer = vec![false; END_KEY as usize];
        for k in map.keys() {
            // Check the key is something we expected
            assert!((START_KEY..END_KEY).contains(&k));
            if found_buffer[k as usize] {
                panic!("Found key two times");
            } else {
                found_buffer[k as usize] = true;
            }
        }
        // Check all the keys where found
        assert!(found_buffer.iter().all(|b| *b));

        // Test iterator over values
        let mut found_buffer = vec![false; END_KEY as usize];
        for v in map.values() {
            let v_as_index = *v as usize;
            if found_buffer[v_as_index] {
                panic!("Found value two times");
            } else {
                found_buffer[v_as_index] = true;
            }
        }
        // Check all the keys where found
        assert!(found_buffer.iter().all(|b| *b));
    }
}
