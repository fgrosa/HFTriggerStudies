def GetMaskOfBits(bits):
    '''
    Helper method to get bit mask from bits

    Arguments
    ----------
    - list of bits

    Returns
    ----------
    - mask corresponding to the input bits
    '''
    mask = 0
    for bit in bits:
        mask += 2**bit

    return mask


def FilterBitDf(dfToFilter, column, bitsToTest, logic='or'):
    '''
    Method to apply selection testing one or more bits

    Arguments
    ----------
    - pandas dataframe to filter
    - colum with bitmap
    - list of bits to test
    - logic to combine the bits (and, or)

    Returns
    ----------
    - filtered pandas dataframe
    '''
    maskOfBits = GetMaskOfBits(bitsToTest)
    flags = dfToFilter[column].astype(int) & maskOfBits
    if logic == 'or':
        flags = flags.astype('bool')
    elif logic == 'and':
        flags -= maskOfBits
        flags = ~flags.astype('bool')
    elif logic == 'not':
        flags = ~flags.astype('bool')
    else:
        print('Error: only and, or, and not logics are supported for bitwise operations')
        return None

    dfFilt = dfToFilter[flags.to_numpy()]

    return dfFilt
